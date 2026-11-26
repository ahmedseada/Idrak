// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using Idrak.Abstraction.Testing;
using Idrak.Imaging;
using Idrak.Vision;

// Plan 38, phase F1: the library's outline tracer "contour" (Idrak.Imaging) on real glyph outlines of the test fonts
// (DejaVu and three Arabic fonts: holes, sharp and smooth turns), drawn through OutlineRasterizers, traced, drawn again
// and compared; and Otsu's threshold moved to ImageThresholds (Idrak.Abstraction), which Idrak.Vision's Foreground.Otsu
// now calls, its results unchanged.
internal abstract partial class Tests
{
    // DejaVu's glyphs: holes (a e g o B 8 %), sharp corners (k M y 2 &) and smooth turns.
    private const string DejaVuGlyphs = "aegoB8%&kMy2";

    // Arabic letters, isolated: loops and holes (heh, sad, tah, waw, meem), sharp and smooth turns (ain, kaf), dots (sheen).
    private static readonly int[] ArabicGlyphs = [0x0647, 0x0635, 0x0637, 0x0648, 0x0645, 0x0639, 0x0643, 0x0634];

    internal static void ContourTracerMeasured(Device device)
    {
        CpuOnly(device);
        var tracer = new ContourTracer();
        var settings = tracer.Read(PageOptions.None);
        var plain = tracer.Read(PageOptionsOf((ContourTracer.SimplifyKey, "0")));

        // A sheet cell's glyphs: 96 pixels per em (a cell of about 120 pixels: a centimetre scanned at 300 dpi), DejaVu's and Amiri's.
        var glyphs = new List<(GlyphCoverage Coverage, OutlinePlacement At)>();
        foreach (var (_, unitsPerEm, outlines) in TracerFonts().Take(2))
        {
            foreach (var (_, outline) in outlines)
            {
                glyphs.Add(Rasterized(outline, unitsPerEm, 96, 0.3f, 0.6f));
            }
        }

        // Cold: the process's first trace, which compiles the tracer's methods (rule 76: reported apart).
        long bytes = GC.GetAllocatedBytesForCurrentThread();
        long start = Stopwatch.GetTimestamp();
        var first = tracer.Trace(glyphs[0].Coverage, glyphs[0].At, settings);
        double cold = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        long coldBytes = GC.GetAllocatedBytesForCurrentThread() - bytes;
        Check(!first.IsEmpty, "the first glyph traced to an outline");

        // Warm: thirty rounds to let the tiered JIT finish, then twenty timed.
        for (int round = 0; round < 30; round++)
        {
            foreach (var (coverage, at) in glyphs)
            {
                tracer.Trace(coverage, at, settings);
            }
        }

        const int Rounds = 20;
        bytes = GC.GetAllocatedBytesForCurrentThread();
        start = Stopwatch.GetTimestamp();
        for (int round = 0; round < Rounds; round++)
        {
            foreach (var (coverage, at) in glyphs)
            {
                tracer.Trace(coverage, at, settings);
            }
        }

        double warm = Stopwatch.GetElapsedTime(start).TotalMicroseconds / (Rounds * glyphs.Count);
        double warmBytes = (GC.GetAllocatedBytesForCurrentThread() - bytes) / (double)(Rounds * glyphs.Count);

        // The outline's own arrays (what a trace must allocate), and the points before and after simplification.
        long before = 0, after = 0, output = 0;
        foreach (var (coverage, at) in glyphs)
        {
            var simplified = tracer.Trace(coverage, at, settings);
            before += tracer.Trace(coverage, at, plain).Points.Length / 2;
            after += simplified.Points.Length / 2;
            output += 24 + Align8(simplified.Verbs.Length) + 24 + Align8(4 * simplified.Points.Length) + 56;
        }

        double outputBytes = output / (double)glyphs.Count;
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"    contour at 96 px per em ({glyphs.Count} glyphs of DejaVu and Amiri): cold {cold:F1} ms and {coldBytes:N0} bytes (the first trace); warm {warm:F0} µs and {warmBytes:N0} bytes a glyph (its outline's arrays {outputBytes:N0}); points per outline {before / (double)glyphs.Count:F0} traced, {after / (double)glyphs.Count:F0} simplified ({before / (double)after:F1}x fewer)"));
        Check(after < before, $"simplified, {after} points; as traced, {before}");
        Check(warmBytes <= outputBytes + 4096, string.Create(CultureInfo.InvariantCulture,
            $"a warm trace allocates its outline and little else: {warmBytes:N0} bytes a glyph, its outline's arrays {outputBytes:N0} (the working buffers are rented and returned)"));

        static long Align8(long n) => (n + 7) / 8 * 8;
    }

    internal static void ContourTracerOptions(Device device)
    {
        CpuOnly(device);
        var tracer = new ContourTracer();
        ThrowsNaming<ArgumentException>(() => tracer.Read(PageOptionsOf(("trace-speed", "1"))), "a key it does not take", "'trace-speed'",
            "trace-threshold, trace-simplify, trace-corner, trace-smoothing");
        ThrowsNaming<ArgumentException>(() => tracer.Read(PageOptionsOf((ContourTracer.ThresholdKey, "1.5"))), "a level past 0.99",
            "trace-threshold is auto (Otsu's level of the coverage) or an ink level from 0.01 to 0.99, not '1.5'");
        ThrowsNaming<ArgumentException>(() => tracer.Read(PageOptionsOf((ContourTracer.ThresholdKey, "0"))), "a level of no ink", "not '0'");
        ThrowsNaming<ArgumentException>(() => tracer.Read(PageOptionsOf((ContourTracer.ThresholdKey, "otsu"))), "a level that is no number", "not 'otsu'");
        ThrowsNaming<ArgumentException>(() => tracer.Read(PageOptionsOf((ContourTracer.SimplifyKey, "-1"))), "a tolerance below 0",
            "trace-simplify (pixels) is a number from 0 to 8, not '-1'");
        ThrowsNaming<ArgumentException>(() => tracer.Read(PageOptionsOf((ContourTracer.CornerKey, "200"))), "a corner past 180 degrees",
            "trace-corner (degrees) is a number from 0 to 180, not '200'");
        ThrowsNaming<ArgumentException>(() => tracer.Read(PageOptionsOf((ContourTracer.SmoothingKey, "0.1"))), "a smoothing under half a pixel",
            "trace-smoothing (pixels) is a number from 0.5 to 16, not '0.1'");

        // The promise: the simplification plus the most the level's edge strays from the half-ink edge ("auto" at 0.1 or 0.9).
        float defaults = tracer.Tolerance(tracer.Read(PageOptions.None));
        float half = tracer.Tolerance(tracer.Read(PageOptionsOf(("TRACE-THRESHOLD", "0.5"), ("Trace-Simplify", "0.25"))));
        float low = tracer.Tolerance(tracer.Read(PageOptionsOf((ContourTracer.ThresholdKey, "0.2"))));
        float auto = tracer.Tolerance(tracer.Read(PageOptionsOf((ContourTracer.ThresholdKey, "Auto"), (ContourTracer.CornerKey, "30"), (ContourTracer.SmoothingKey, "4"))));
        Check(MathF.Abs(defaults - (ContourTracer.DefaultSimplify + ContourTracer.EdgeReach(null))) < 1e-6f && MathF.Abs(ContourTracer.EdgeReach(null) - 1.02f) < 1e-5f
              && MathF.Abs(half - 0.35f) < 1e-5f && MathF.Abs(low - (0.5f + 0.1f + 0.45f + 0.18f)) < 1e-5f && auto == defaults
              && MathF.Abs(ContourTracer.EdgeReach(0.9f) - ContourTracer.EdgeReach(null)) < 1e-5f && ContourTracer.EdgeReach(0.6f) < ContourTracer.EdgeReach(0.7f),
            string.Create(CultureInfo.InvariantCulture, $"the tolerances: defaults {defaults}, at the half with a quarter pixel {half}, at 0.2 {low}, auto {auto}"));

        // No placement is refused naming it; no ink, or ink below the level, is no outline.
        var settings = tracer.Read(PageOptions.None);
        var ink = new GlyphCoverage([0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f], 3, 3);
        ThrowsNaming<ArgumentException>(() => tracer.Trace(ink, new OutlinePlacement(0f, 1f, 2f), settings), "a placement of no scale", "Not a placement", "scale 0");
        ThrowsNaming<ArgumentException>(() => tracer.Trace(ink, new OutlinePlacement(1f, float.NaN, 2f), settings), "a placement not finite", "origin (NaN, 2)");
        Check(tracer.Trace(GlyphCoverage.Blank(8, 8), new OutlinePlacement(1f, 0f, 8f), settings).IsEmpty, "no ink: no outline");
        var faint = new float[64];
        Array.Fill(faint, 0.05f);
        Check(tracer.Trace(new GlyphCoverage(faint, 8, 8), new OutlinePlacement(1f, 0f, 8f), settings).IsEmpty, "the same faint ink everywhere, below Otsu's level: no outline");
        var dot = tracer.Trace(ink, new OutlinePlacement(0.5f, 0f, 3f), tracer.Read(PageOptionsOf((ContourTracer.ThresholdKey, "0.5"))));
        var (minX, minY, maxX, maxY) = dot.Bounds();
        Check(!dot.IsEmpty && dot.Verbs.Span[0] == OutlineVerb.Move && dot.Verbs.Span[^1] == OutlineVerb.Close && minX >= 1f && maxX <= 5f && minY >= 1f && maxY <= 5f,
            string.Create(CultureInfo.InvariantCulture, $"one inked pixel at (1, 1), two units a pixel: a contour round it within half a pixel, ({minX}, {minY}) to ({maxX}, {maxY})"));
    }

    internal static void ContourTracerRoundTrip(Device device)
    {
        CpuOnly(device);
        var tracer = new ContourTracer();
        var fonts = TracerFonts();
        var plain = tracer.Read(PageOptionsOf((ContourTracer.SimplifyKey, "0")));
        long checkedCentres = 0, skipped = 0;
        var lines = new List<string>();
        foreach (var options in new[] { PageOptions.None, PageOptionsOf((ContourTracer.ThresholdKey, "0.5")), PageOptionsOf((ContourTracer.SimplifyKey, "1"), (ContourTracer.CornerKey, "45")) })
        {
            var settings = tracer.Read(options);
            float tolerance = tracer.Tolerance(settings);
            string given = options.Count == 0 ? "defaults" : string.Join(", ", options.Select(o => $"{o.Key}={o.Value}"));
            foreach (float em in new[] { 24f, 48f, 96f })
            {
                double difference = 0, length = 0;
                long before = 0, after = 0;
                int index = 0;
                foreach (var (font, unitsPerEm, glyphs) in fonts)
                {
                    foreach (var (name, outline) in glyphs)
                    {
                        float dx = 0.13f + 0.29f * (index % 3), dy = 0.71f - 0.23f * (index % 4);
                        index++;
                        var (coverage, at) = Rasterized(outline, unitsPerEm, em, dx, dy);
                        var traced = tracer.Trace(coverage, at, settings);
                        string what = string.Create(CultureInfo.InvariantCulture, $"{font} {name} at {em} px per em ({given})");
                        Check(!traced.IsEmpty && !traced.Verbs.Span.Contains(OutlineVerb.Cubic), $"{what}: an outline of lines and quadratic curves");

                        // Every pixel centre: inside the outline exactly when its coverage reaches the level, one winding.
                        var values = coverage.Values.Span;
                        float level = settings.Options.ContainsKey(ContourTracer.ThresholdKey) ? 0.5f
                            : Math.Clamp(ImageThresholds.Otsu(values), ContourTracer.AutoLeast, ContourTracer.AutoMost);
                        var windings = Windings(traced, at, coverage.Width, coverage.Height);
                        int sign = 0;
                        for (int i = 0; i < values.Length; i++)
                        {
                            var (winding, near) = windings[i];
                            if (near || MathF.Abs(values[i] - level) < 1e-5f)
                            {
                                skipped++;
                                continue;
                            }

                            checkedCentres++;
                            bool inside = winding != 0;
                            if (Math.Abs(winding) > 1 || (sign != 0 && winding != 0 && winding != sign) || inside != values[i] >= level)
                            {
                                Check(false, string.Create(CultureInfo.InvariantCulture,
                                    $"{what}: pixel ({i % coverage.Width}, {i / coverage.Width}) of coverage {values[i]} (the level {level}) winds {winding}{(sign != 0 ? $", other ink {sign}" : "")}"));
                            }

                            sign = winding != 0 ? winding : sign;
                        }

                        // Drawn again: the coverage differs by at most the promised tolerance times the outline's length.
                        var again = Filled(traced, at, coverage.Width, coverage.Height);
                        double glyphDifference = 0;
                        for (int i = 0; i < values.Length; i++)
                        {
                            glyphDifference += Math.Abs(values[i] - again[i]);
                        }

                        double glyphLength = OutlineLength(outline) * at.Scale;
                        Check(glyphDifference <= tolerance * glyphLength, string.Create(CultureInfo.InvariantCulture,
                            $"{what}: drawn again, the coverage differs by {glyphDifference:F2} pixels of ink along {glyphLength:F1} pixels of edge, more than the promised {tolerance} pixels"));
                        difference += glyphDifference;
                        length += glyphLength;

                        // Holes kept, turning the other way: DejaVu's contours, large enough to keep theirs, by direction.
                        if (em >= 96 && index <= DejaVuGlyphs.Length)
                        {
                            var (outer, holes) = Directions(outline);
                            var (tracedOuter, tracedHoles) = Directions(traced);
                            Check(tracedOuter == outer && tracedHoles == holes, $"{what}: {tracedOuter} contours clockwise and {tracedHoles} the other way; the glyph has {outer} and {holes}");
                        }

                        before += tracer.Trace(coverage, at, plain).Points.Length / 2;
                        after += traced.Points.Length / 2;
                    }
                }

                Check(after < before, $"{em} px per em ({given}): {after} points simplified, {before} as traced");
                lines.Add(string.Create(CultureInfo.InvariantCulture,
                    $"{given}, {em} px per em: drawn again, {difference / length:F3} pixels of ink per pixel of edge (promised at most {tolerance:F2}); {before / (double)index:F0} points a glyph as traced, {after / (double)index:F0} simplified"));
            }
        }

        Check(skipped * 100 <= checkedCentres, $"{skipped} pixel centres on the edge itself left out of {checkedCentres + skipped}");
        Console.WriteLine($"    contour, {checkedCentres:N0} pixel centres each inside exactly when its coverage reaches the level ({skipped:N0} on the edge left out):");
        foreach (string line in lines)
        {
            Console.WriteLine("      " + line);
        }
    }

    internal static void OtsuMovedUnchanged(Device device)
    {
        CpuOnly(device);
        var random = new Random(38);
        var cases = new List<(string Name, float[] Values)>
        {
            ("no values", []), ("one value", [0.4f]), ("blank", new float[64]), ("all ink", Enumerable.Repeat(1f, 64).ToArray()),
            ("two levels", [0.1f, 0.12f, 0.08f, 0.9f, 0.88f, 0.92f]),
        };
        for (int k = 0; k < 24; k++)
        {
            var values = new float[1 + random.Next(4000)];
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = (k % 3) switch
                {
                    0 => (float)random.NextDouble(),                                                   // uniform
                    1 => random.Next(256) / 255f,                                                      // 8-bit grey
                    _ => Math.Clamp((random.Next(3) == 0 ? 0.2f : 0.85f) + 0.1f * (float)(random.NextDouble() - 0.5), 0f, 1f),   // ink and paper with noise
                };
            }

            cases.Add(($"random {k}", values));
        }

        var canvases = new[]
        {
            Canvas(40, 30, (2, 2, 5, 5, false), (15, 5, 1, 1, false), (16, 6, 1, 1, false), (30, 25, 1, 1, false)),
            Canvas(50, 40, (20, 10, 9, 14, true)),
            Canvas(80, 40, (5, 3, 16, 16, true), (40, 4, 5, 20, false), (60, 20, 14, 14, false)),
        };
        foreach (var canvas in canvases)
        {
            cases.Add(($"a {canvas.Width} x {canvas.Height} canvas", Foreground.Grey(canvas)));
        }

        foreach (var (name, values) in cases)
        {
            int expected = BitConverter.SingleToInt32Bits(OtsuAsLoops(values));
            Check(BitConverter.SingleToInt32Bits(ImageThresholds.Otsu(values)) == expected && BitConverter.SingleToInt32Bits(Foreground.Otsu(values)) == expected,
                $"{name}: ImageThresholds.Otsu {ImageThresholds.Otsu(values):R}, Foreground.Otsu {Foreground.Otsu(values):R}, the method as loops {OtsuAsLoops(values):R}");
        }

        // Where Vision uses it: a page's foreground, dark on light, inverted and cut at Otsu's level of the inverted grey.
        foreach (var canvas in canvases)
        {
            var grey = Foreground.Grey(canvas);
            for (int i = 0; i < grey.Length; i++)
            {
                grey[i] = 1f - grey[i];
            }

            var foreground = Foreground.Extract(canvas);
            Check(foreground.Inverted && BitConverter.SingleToInt32Bits(foreground.Threshold) == BitConverter.SingleToInt32Bits(OtsuAsLoops(grey)),
                $"the {canvas.Width} x {canvas.Height} canvas's foreground: threshold {foreground.Threshold:R}, the method as loops {OtsuAsLoops(grey):R}");
        }
    }

    // Otsu's method written out from its definition: for each level the groups below and above it counted and summed
    // afresh from the histogram, the level of the largest between-group variance (the first of equal ones), at least 0.1.
    private static float OtsuAsLoops(float[] values)
    {
        var histogram = new long[256];
        foreach (float v in values)
        {
            histogram[Math.Clamp((int)(v * 255f), 0, 255)]++;
        }

        double total = values.Length, sum = 0;
        for (int i = 0; i < 256; i++)
        {
            sum += i * (double)histogram[i];
        }

        double best = -1;
        int level = 128;
        for (int i = 0; i < 256; i++)
        {
            long below = 0;
            double belowSum = 0;
            for (int j = 0; j <= i; j++)
            {
                below += histogram[j];
                belowSum += j * (double)histogram[j];
            }

            if (below == 0 || below == total)
            {
                continue;
            }

            double meanBelow = belowSum / below, meanAbove = (sum - belowSum) / (total - below);
            double between = below * (total - below) * (meanBelow - meanAbove) * (meanBelow - meanAbove);
            if (between > best)
            {
                (best, level) = (between, i);
            }
        }

        return Math.Max((level + 1) / 255f, 0.1f);
    }

    // The test fonts' glyphs for the tracer: DejaVu's, and the Arabic letters of Amiri, Reem Kufi and Cairo; each font its own em.
    private static List<(string Font, int UnitsPerEm, List<(string Name, GlyphOutline Outline)> Glyphs)> TracerFonts()
    {
        var dejavu = SubsetFace();
        var fonts = new List<(string, int, List<(string, GlyphOutline)>)>
        {
            ("DejaVu Sans", dejavu.Metrics.UnitsPerEm, [.. DejaVuGlyphs.Select(c => (c.ToString(), dejavu.Outline(dejavu.Glyph(c))))]),
        };
        foreach (string file in new[] { "amiri", "reem-kufi", "cairo" })
        {
            var face = new OpenTypeFormat().Read(File.ReadAllBytes(FontData($"arabic/{file}.ttf")), file).Single();
            fonts.Add((face.Family, face.Metrics.UnitsPerEm, [.. ArabicGlyphs.Select(code => ($"U+{code:X4}", face.Outline(face.Glyph(code))))]));
        }

        foreach (var (font, _, glyphs) in fonts)
        {
            Check(glyphs.All(g => !g.Item2.IsEmpty), $"{font} has every glyph the tracer is tested on");
        }

        return fonts;
    }

    // A glyph's outline filled at `em` pixels per em, (dx, dy) off the pixel grid, with four pixels of paper round it.
    private static (GlyphCoverage Coverage, OutlinePlacement At) Rasterized(GlyphOutline outline, int unitsPerEm, float em, float dx, float dy)
    {
        const int Margin = 4;
        float scale = em / unitsPerEm;
        var (minX, minY, maxX, maxY) = outline.Bounds();
        int width = (int)MathF.Ceiling((maxX - minX) * scale) + 2 * Margin + 1, height = (int)MathF.Ceiling((maxY - minY) * scale) + 2 * Margin + 1;
        var at = new OutlinePlacement(scale, Margin - minX * scale + dx, Margin + maxY * scale + dy);
        return (new GlyphCoverage(Filled(outline, at, width, height), width, height), at);
    }

    private static float[] Filled(GlyphOutline outline, OutlinePlacement at, int width, int height)
    {
        var values = new float[width * height];
        new CoverageRasterizer().Fill(outline, at.Scale, at.X, at.Y, values, width, height);
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = Math.Clamp(values[i], 0f, 1f);
        }

        return values;
    }

    // How the outline winds round each pixel centre, worked in font units (pixels map to them monotonically, so a crossing
    // keeps its side of a centre): the crossings of the centre's row to its left, a part counted from its start's row up
    // to, not with, its end's (in pixels; y is flipped in units); and whether the edge passes the centre too closely for
    // rounding to decide (a crossing within a thousandth of a pixel, or a point of the outline within a twentieth).
    private static (int Winding, bool Near)[] Windings(GlyphOutline outline, OutlinePlacement at, int width, int height)
    {
        var parts = UnitParts(outline);
        var windings = new (int, bool)[width * height];
        var crossings = new List<(float U, int Direction)>();
        var ends = new List<(float U, float V)>();
        float nearU = 1e-3f / at.Scale, nearEnd = 0.05f / at.Scale;
        for (int row = 0; row < height; row++)
        {
            float vs = at.ToUnits(0.5f, row + 0.5f).Y;
            crossings.Clear();
            ends.Clear();
            foreach (var part in parts)
            {
                float lo = MathF.Min(part.V0, part.V1), hi = MathF.Max(part.V0, part.V1);
                if (lo < vs && vs <= hi)
                {
                    crossings.Add((CrossingU(part, vs), part.V1 < part.V0 ? 1 : -1));
                }

                if (MathF.Abs(part.V0 - vs) < nearEnd)
                {
                    ends.Add((part.U0, part.V0));
                }
            }

            crossings.Sort((a, b) => a.U.CompareTo(b.U));
            for (int col = 0; col < width; col++)
            {
                var (us, _) = at.ToUnits(col + 0.5f, row + 0.5f);
                int winding = 0;
                bool near = false;
                foreach (var (u, direction) in crossings)
                {
                    winding += u < us ? direction : 0;
                    near |= MathF.Abs(u - us) < nearU;
                }

                foreach (var (u, v) in ends)
                {
                    near |= (u - us) * (u - us) + (v - vs) * (v - vs) < nearEnd * nearEnd;
                }

                windings[row * width + col] = (winding, near);
            }
        }

        return windings;
    }

    // A part of an outline that runs one way in v: a line, or a quadratic curve (S, Q, E) between parameters T0 and T1.
    private readonly record struct UnitPart(float U0, float V0, float U1, float V1, bool Curve, float Su, float Sv, float Qu, float Qv, float Eu, float Ev, float T0, float T1);

    private static List<UnitPart> UnitParts(GlyphOutline outline)
    {
        var parts = new List<UnitPart>();
        var verbs = outline.Verbs.Span;
        var p = outline.Points.Span;
        float su = 0, sv = 0, cu = 0, cv = 0;
        int i = 0;
        foreach (var verb in verbs)
        {
            switch (verb)
            {
                case OutlineVerb.Move:
                    Close();
                    (su, sv, cu, cv) = (p[i], p[i + 1], p[i], p[i + 1]);
                    i += 2;
                    break;
                case OutlineVerb.Line:
                    Line(p[i], p[i + 1]);
                    i += 2;
                    break;
                case OutlineVerb.Quadratic:
                {
                    float qu = p[i], qv = p[i + 1], eu = p[i + 2], ev = p[i + 3];
                    float bend = cv - 2 * qv + ev, turn = bend != 0 ? (cv - qv) / bend : -1f;
                    float[] cuts = turn > 0 && turn < 1 ? [0f, turn, 1f] : [0f, 1f];
                    for (int k = 0; k + 1 < cuts.Length; k++)
                    {
                        var (u0, v0) = Bezier(cu, cv, qu, qv, eu, ev, cuts[k]);
                        var (u1, v1) = Bezier(cu, cv, qu, qv, eu, ev, cuts[k + 1]);
                        parts.Add(new UnitPart(u0, v0, u1, v1, true, cu, cv, qu, qv, eu, ev, cuts[k], cuts[k + 1]));
                    }

                    (cu, cv) = (eu, ev);
                    i += 4;
                    break;
                }

                case OutlineVerb.Cubic:
                    throw new InvalidOperationException("The tracer's outline holds a cubic curve.");
                default:
                    Close();
                    break;
            }
        }

        Close();
        return parts;

        void Line(float u, float v)
        {
            parts.Add(new UnitPart(cu, cv, u, v, false, 0, 0, 0, 0, 0, 0, 0, 1));
            (cu, cv) = (u, v);
        }

        void Close()
        {
            if (cu != su || cv != sv)
            {
                Line(su, sv);
            }
        }
    }

    private static (float U, float V) Bezier(float su, float sv, float qu, float qv, float eu, float ev, float t)
    {
        float r = 1 - t;
        return (r * r * su + 2 * r * t * qu + t * t * eu, r * r * sv + 2 * r * t * qv + t * t * ev);
    }

    private static float CrossingU(in UnitPart part, float vs)
    {
        if (!part.Curve)
        {
            return part.U0 + (vs - part.V0) * (part.U1 - part.U0) / (part.V1 - part.V0);
        }

        double a = part.Sv - 2.0 * part.Qv + part.Ev, b = 2.0 * (part.Qv - part.Sv), c = part.Sv - (double)vs, t;
        if (Math.Abs(a) < 1e-12)
        {
            t = -c / b;
        }
        else
        {
            double root = Math.Sqrt(Math.Max(0, b * b - 4 * a * c)), first = (-b - root) / (2 * a), second = (-b + root) / (2 * a);
            t = first >= part.T0 - 1e-9 && first <= part.T1 + 1e-9 ? first : second;
        }

        t = Math.Clamp(t, part.T0, part.T1);
        double r = 1 - t;
        return (float)(r * r * part.Su + 2 * r * t * part.Qu + t * t * part.Eu);
    }

    // An outline's contours as polylines in font units, each curve in 32 steps.
    private static List<List<(float X, float Y)>> Polylines(GlyphOutline outline)
    {
        const int Steps = 32;
        var contours = new List<List<(float X, float Y)>>();
        var verbs = outline.Verbs.Span;
        var p = outline.Points.Span;
        List<(float X, float Y)>? current = null;
        int i = 0;
        foreach (var verb in verbs)
        {
            switch (verb)
            {
                case OutlineVerb.Move:
                    current = [(p[i], p[i + 1])];
                    contours.Add(current);
                    i += 2;
                    break;
                case OutlineVerb.Line:
                    current!.Add((p[i], p[i + 1]));
                    i += 2;
                    break;
                case OutlineVerb.Quadratic:
                {
                    var (sx, sy) = current![^1];
                    for (int k = 1; k <= Steps; k++)
                    {
                        current.Add(Bezier(sx, sy, p[i], p[i + 1], p[i + 2], p[i + 3], k / (float)Steps));
                    }

                    i += 4;
                    break;
                }

                case OutlineVerb.Cubic:
                {
                    var (sx, sy) = current![^1];
                    for (int k = 1; k <= Steps; k++)
                    {
                        float t = k / (float)Steps, r = 1 - t;
                        float a = r * r * r, b = 3 * r * r * t, c = 3 * r * t * t, d = t * t * t;
                        current.Add((a * sx + b * p[i] + c * p[i + 2] + d * p[i + 4], a * sy + b * p[i + 1] + c * p[i + 3] + d * p[i + 5]));
                    }

                    i += 6;
                    break;
                }
            }
        }

        return contours;
    }

    // The length of an outline's edge, in font units.
    private static double OutlineLength(GlyphOutline outline)
    {
        double length = 0;
        foreach (var contour in Polylines(outline))
        {
            for (int k = 0; k < contour.Count; k++)
            {
                var (x0, y0) = contour[k];
                var (x1, y1) = contour[(k + 1) % contour.Count];
                length += Math.Sqrt((double)(x1 - x0) * (x1 - x0) + (double)(y1 - y0) * (y1 - y0));
            }
        }

        return length;
    }

    // How many contours turn clockwise (outer, in font units) and how many the other way (holes).
    private static (int Clockwise, int Other) Directions(GlyphOutline outline)
    {
        int clockwise = 0, other = 0;
        foreach (var contour in Polylines(outline))
        {
            double area = 0;
            for (int k = 0; k < contour.Count; k++)
            {
                var (x0, y0) = contour[k];
                var (x1, y1) = contour[(k + 1) % contour.Count];
                area += (double)x0 * y1 - (double)x1 * y0;
            }

            if (area < 0)
            {
                clockwise++;
            }
            else if (area > 0)
            {
                other++;
            }
        }

        return (clockwise, other);
    }
}
