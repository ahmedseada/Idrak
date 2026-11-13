// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Text.Json;

namespace Idrak.Imaging;

/// <summary>What a font's glyph measures: its ink box, its strokes' thickness and its ink's top and bottom around its middle, in font units (y up).</summary>
internal readonly record struct InkStats(double MinX, double MinY, double MaxX, double MaxY, double Thickness, double CenterX, double Top, double Bottom)
{
    public double Width => MaxX - MinX;

    public double Height => MaxY - MinY;
}

/// <summary>
/// Measures glyphs by drawing them with the registered outline rasterizer at a fixed resolution and reading the ink (the
/// coverage above a threshold), the way a person measures a printed glyph: its ink box, its strokes' typical thickness
/// (the median of each ink pixel's shorter run, across or down) and the top and bottom of its ink within a window around
/// its middle (where a mark is placed).
/// </summary>
internal static class InkMeasure
{
    /// <summary>The resolution the ink is read at: pixels per em.</summary>
    public const double PixelsPerEm = 256;

    private const float Threshold = 110f / 255f;

    /// <summary>The ink of <paramref name="outline"/>, or null when it has none.</summary>
    /// <param name="outline">The glyph's outline.</param>
    /// <param name="unitsPerEm">The font's units per em.</param>
    /// <param name="window">Half the width, in font units around the ink's middle, where its top and bottom are read.</param>
    public static InkStats? Of(GlyphOutline outline, int unitsPerEm, double window)
    {
        if (outline.IsEmpty)
        {
            return null;
        }

        var (bx0, by0, bx1, by1) = outline.Bounds();
        double scale = PixelsPerEm / unitsPerEm;
        int width = (int)Math.Ceiling((bx1 - bx0) * scale) + 4, height = (int)Math.Ceiling((by1 - by0) * scale) + 4;
        float ox = (float)(-bx0 * scale + 2), oy = (float)(by1 * scale + 2);
        int size = width * height;
        float[] coverage = ArrayPool<float>.Shared.Rent(size);
        int[] across = ArrayPool<int>.Shared.Rent(size);
        int[] down = ArrayPool<int>.Shared.Rent(size);
        int[] histogram = ArrayPool<int>.Shared.Rent(Math.Max(width, height) + 1);
        try
        {
            var ink = coverage.AsSpan(0, size);
            ink.Clear();
            OutlineRasterizers.Get().Fill(outline, (float)scale, ox, oy, ink, width, height);
            int minCol = int.MaxValue, maxCol = -1, minRow = int.MaxValue, maxRow = -1;

            // Runs across, row by row (each ink pixel gets its run's length), and runs down: the run above it so far, then
            // completed from the bottom up (rows walked in memory order both ways).
            for (int y = 0; y < height; y++)
            {
                int row = y * width;
                for (int x = 0; x < width;)
                {
                    if (ink[row + x] <= Threshold)
                    {
                        across[row + x] = 0;
                        x++;
                        continue;
                    }

                    int start = x;
                    while (x < width && ink[row + x] > Threshold)
                    {
                        x++;
                    }

                    for (int k = start; k < x; k++)
                    {
                        across[row + k] = x - start;
                    }

                    minCol = Math.Min(minCol, start);
                    maxCol = Math.Max(maxCol, x - 1);
                    minRow = Math.Min(minRow, y);
                    maxRow = Math.Max(maxRow, y);
                }

                for (int x = 0; x < width; x++)
                {
                    down[row + x] = ink[row + x] > Threshold ? (y > 0 ? down[row - width + x] : 0) + 1 : 0;
                }
            }

            if (maxCol < 0)
            {
                return null;
            }

            for (int y = height - 2; y >= 0; y--)
            {
                int row = y * width;
                for (int x = 0; x < width; x++)
                {
                    if (down[row + x] > 0 && down[row + width + x] > down[row + x])
                    {
                        down[row + x] = down[row + width + x];       // the whole run's length, carried up from its bottom
                    }
                }
            }

            histogram.AsSpan(0, Math.Max(width, height) + 1).Clear();
            int count = 0;
            for (int i = 0; i < size; i++)
            {
                if (across[i] > 0)
                {
                    histogram[Math.Min(across[i], down[i])]++;
                    count++;
                }
            }

            int median = 0;
            for (int seen = 0; median < histogram.Length && seen + histogram[median] <= count / 2; median++)
            {
                seen += histogram[median];
            }

            double center = (minCol + maxCol + 1) / 2.0;
            double reach = window * scale;
            int top = int.MaxValue, bottom = -1;
            for (int y = minRow; y <= maxRow; y++)
            {
                int row = y * width;
                for (int x = Math.Max(minCol, (int)Math.Floor(center - reach)); x <= Math.Min(maxCol, (int)Math.Ceiling(center + reach)); x++)
                {
                    if (across[row + x] > 0)
                    {
                        top = Math.Min(top, y);
                        bottom = Math.Max(bottom, y);
                        break;
                    }
                }
            }

            if (bottom < 0)
            {
                (top, bottom) = (minRow, maxRow);
            }

            return new InkStats((minCol - ox) / scale, (oy - (maxRow + 1)) / scale, (maxCol + 1 - ox) / scale, (oy - minRow) / scale, median / scale,
                (center - ox) / scale, (oy - top) / scale, (oy - bottom) / scale);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(coverage);
            ArrayPool<int>.Shared.Return(across);
            ArrayPool<int>.Shared.Return(down);
            ArrayPool<int>.Shared.Return(histogram);
        }
    }

    /// <summary>The median of <paramref name="values"/> (0 when empty).</summary>
    public static double Median(List<double> values)
    {
        if (values.Count == 0)
        {
            return 0;
        }

        values.Sort();
        int mid = values.Count / 2;
        return values.Count % 2 == 1 ? values[mid] : (values[mid - 1] + values[mid]) / 2;
    }
}

/// <summary>One stroke of a drawn glyph, as the data writes it: a Catmull-Rom spline, a sharp polyline or an elliptic arc.</summary>
internal sealed record DrawnStroke(DrawnStrokeKind Kind, double[] Values, int Samples, double Shrink)
{
    /// <summary>The stroke's centre line on the unit box (u across, v up), sampled as the data says.</summary>
    public List<(double U, double V)> Path()
    {
        var v = Values;
        var path = new List<(double, double)>();
        switch (Kind)
        {
            case DrawnStrokeKind.Spline:
            {
                // Catmull-Rom through the points, the end points repeated, Samples per span.
                int n = v.Length / 2;
                (double X, double Y) P(int i) => (v[2 * Math.Clamp(i, 0, n - 1)], v[2 * Math.Clamp(i, 0, n - 1) + 1]);
                for (int i = 0; i < n - 1; i++)
                {
                    var (p0, p1, p2, p3) = (P(i - 1), P(i), P(i + 1), P(i + 2));
                    for (int k = 0; k < Samples; k++)
                    {
                        double t = (double)k / Samples;
                        path.Add((CatmullRom(p0.X, p1.X, p2.X, p3.X, t), CatmullRom(p0.Y, p1.Y, p2.Y, p3.Y, t)));
                    }
                }

                path.Add((v[^2], v[^1]));
                break;
            }

            case DrawnStrokeKind.Poly:
                for (int i = 0; i + 3 < v.Length; i += 2)
                {
                    for (int k = 0; k < Samples; k++)
                    {
                        double t = (double)k / Samples;
                        path.Add((v[i] + (v[i + 2] - v[i]) * t, v[i + 1] + (v[i + 3] - v[i + 1]) * t));
                    }
                }

                path.Add((v[^2], v[^1]));
                break;

            default:
                // An elliptic arc: centre, radii, start and end angles in degrees; the radii shrink along it by Shrink.
                for (int i = 0; i <= Samples; i++)
                {
                    double f = (double)i / Samples, t = (v[4] + (v[5] - v[4]) * f) * Math.PI / 180, k = 1 - Shrink * f;
                    path.Add((v[0] + v[2] * k * PortableMath.Cos(t), v[1] + v[3] * k * PortableMath.Sin(t)));
                }

                break;
        }

        return path;
    }

    private static double CatmullRom(double p0, double p1, double p2, double p3, double t) =>
        0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t);
}

internal enum DrawnStrokeKind
{
    Spline,
    Poly,
    Ellipse,
}

/// <summary>A glyph the data draws for a character, and where a mark of it sits.</summary>
internal sealed record DrawnGlyph(int CodePoint, FontMarkPlace Place, DrawnStroke[] Strokes);

/// <summary>A set of drawn glyphs: base glyphs (sized like the font's glyphs of <see cref="Measure"/>) or marks.</summary>
/// <param name="Name">The set's name.</param>
/// <param name="Mark">Whether its glyphs are marks (no width, the mark class, placed on their base).</param>
/// <param name="Measure">Which of the font's glyphs size it: "digits" (the map's measured characters), "letters" or "marks".</param>
/// <param name="Glyphs">Its glyphs, in the data's order (the order the hand's wobble is drawn in).</param>
internal sealed record DrawnSet(string Name, bool Mark, string Measure, DrawnGlyph[] Glyphs);

/// <summary>
/// One form of a composed letter: another letter's glyph in a form (its own form when <see cref="FromForm"/> is null),
/// its dots replaced by <see cref="Dots"/> copies of the font's own dot (null: kept), and strokes drawn above or below it.
/// </summary>
internal sealed record ComposedForm(int From, string? FromForm, int? Dots, string Arrange, DrawnStroke[] Strokes, FontMarkPlace Place)
{
    /// <summary>Whether the body's separate marks are left out: contours wound as its main one and lying inside its box (a kaf's inner mark), holes kept.</summary>
    public bool Strip { get; init; }

    /// <summary>Whether a bar parallel to the body's top stroke is added above it, its length, slant and pen measured from that stroke.</summary>
    public bool Bar { get; init; }
}

/// <summary>A letter composed from the font's own glyphs: the same recipe in every form, or one per form.</summary>
internal sealed record ComposedLetter(int CodePoint, ComposedForm? Every, IReadOnlyDictionary<string, ComposedForm> Forms)
{
    /// <summary>The recipe of <paramref name="form"/>, or null when the letter is not composed in it.</summary>
    public ComposedForm? Of(string form) => Forms.TryGetValue(form, out var f) ? f : Every;
}

/// <summary>
/// The library's drawn glyphs (data shipped with Idrak.Imaging, drawn-glyphs.json): sets of glyphs, each a list of pen
/// strokes on a unit box, that a mapped font lacking them gets, drawn at the font's own measured size and weight. The C#
/// knows only the stroke kinds; which characters and shapes is data.
/// </summary>
internal sealed class DrawnGlyphData
{
    private static readonly string[] MeasureKinds = ["digits", "letters", "marks"];

    private static readonly Lazy<DrawnGlyphData> LibraryData = new(() =>
    {
        const string name = "Idrak.Imaging.drawn-glyphs.json";
        using var stream = typeof(DrawnGlyphData).Assembly.GetManifestResourceStream(name)
            ?? throw new InvalidOperationException($"The library's drawn glyphs ({name}) are missing from Idrak.Imaging.");
        var bytes = new byte[stream.Length];
        stream.ReadExactly(bytes);
        return Parse(bytes, "Idrak.Imaging/" + name);
    });

    private static readonly string[] Arrangements = ["row", "up", "down"];

    private DrawnGlyphData(int discSides, DrawnSet[] sets, ComposedLetter[] composed) => (DiscSides, Sets, Composed) = (discSides, sets, composed);

    /// <summary>The library's drawn glyphs, read once.</summary>
    public static DrawnGlyphData Library => LibraryData.Value;

    /// <summary>The sides of the disc drawn at every other point of a stroke (its round ends and joints).</summary>
    public int DiscSides { get; }

    /// <summary>The sets, in order.</summary>
    public DrawnSet[] Sets { get; }

    /// <summary>The letters composed from a font's own glyphs when it lacks them, in order.</summary>
    public ComposedLetter[] Composed { get; }

    /// <summary>Reads drawn glyphs from their JSON; <paramref name="source"/> names them in messages.</summary>
    public static DrawnGlyphData Parse(ReadOnlyMemory<byte> utf8, string source)
    {
        using var document = JsonDocument.Parse(utf8, new JsonDocumentOptions { CommentHandling = JsonCommentHandling.Skip, AllowTrailingCommas = true });
        var root = document.RootElement;
        InvalidDataException Bad(string key, string what) => new($"{source}: '{key}' {what}.");
        int sides = root.TryGetProperty("discSides", out var d) && d.TryGetInt32(out int s) && s >= 3 ? s : 10;
        var sets = new List<DrawnSet>();
        foreach (var set in root.GetProperty("sets").EnumerateArray())
        {
            string name = set.GetProperty("name").GetString() ?? throw Bad("sets.name", "is missing");
            string kind = set.GetProperty("kind").GetString() ?? "";
            if (kind is not ("base" or "mark"))
            {
                throw Bad($"{name}.kind", $"is '{kind}' (valid: base, mark)");
            }

            string measure = set.GetProperty("measure").GetString() ?? "";
            if (Array.IndexOf(MeasureKinds, measure) < 0)
            {
                throw Bad($"{name}.measure", $"is '{measure}' (valid: {string.Join(", ", MeasureKinds)})");
            }

            var glyphs = new List<DrawnGlyph>();
            foreach (var g in set.GetProperty("glyphs").EnumerateObject())
            {
                string key = $"{name}.{g.Name}";
                if (!FontMap.TryCodePoint(g.Name, out int code))
                {
                    throw Bad(key, "is not a character as U+XXXX");
                }

                var place = FontMarkPlace.None;
                if (g.Value.TryGetProperty("place", out var p))
                {
                    place = p.GetString() switch { "above" => FontMarkPlace.Above, "below" => FontMarkPlace.Below, "none" => FontMarkPlace.None, _ => throw Bad($"{key}.place", "is not above, below or none") };
                }

                var strokes = new List<DrawnStroke>();
                foreach (var stroke in g.Value.GetProperty("strokes").EnumerateArray())
                {
                    strokes.Add(Stroke(stroke, key, Bad));
                }

                glyphs.Add(new DrawnGlyph(code, place, [.. strokes]));
            }

            sets.Add(new DrawnSet(name, kind == "mark", measure, [.. glyphs]));
        }

        var composed = new List<ComposedLetter>();
        if (root.TryGetProperty("compose", out var compose))
        {
            foreach (var letter in compose.EnumerateObject())
            {
                string key = $"compose.{letter.Name}";
                if (!FontMap.TryCodePoint(letter.Name, out int code))
                {
                    throw Bad(key, "is not a character as U+XXXX");
                }

                var forms = new Dictionary<string, ComposedForm>(StringComparer.Ordinal);
                if (letter.Value.TryGetProperty("forms", out var perForm))
                {
                    foreach (var f in perForm.EnumerateObject())
                    {
                        forms[FontMap.Forms.Contains(f.Name) ? f.Name : throw Bad($"{key}.forms.{f.Name}", $"is not a form ({string.Join(", ", FontMap.Forms)})")] =
                            Recipe(f.Value, $"{key}.forms.{f.Name}", Bad);
                    }
                }

                var every = letter.Value.TryGetProperty("from", out _) ? Recipe(letter.Value, key, Bad) : null;
                composed.Add(every is null && forms.Count == 0 ? throw Bad(key, "needs \"from\" or \"forms\"") : new ComposedLetter(code, every, forms));
            }
        }

        return new DrawnGlyphData(sides, [.. sets], [.. composed]);
    }

    private static ComposedForm Recipe(JsonElement e, string key, Func<string, string, InvalidDataException> bad)
    {
        string from = e.TryGetProperty("from", out var f) ? f.GetString() ?? "" : throw bad($"{key}.from", "is missing");
        int dot = from.IndexOf('.', StringComparison.Ordinal);
        string? form = dot < 0 ? null : from[(dot + 1)..];
        if (!FontMap.TryCodePoint(dot < 0 ? from : from[..dot], out int code) || (form is not null && !FontMap.Forms.Contains(form)))
        {
            throw bad($"{key}.from", $"is '{from}': a letter as U+XXXX, or U+XXXX.form");
        }

        int? dots = e.TryGetProperty("dots", out var d) ? d.TryGetInt32(out int n) && n is >= 0 and <= 4 ? n : throw bad($"{key}.dots", "must be 0 to 4") : null;
        string arrange = e.TryGetProperty("arrange", out var a) ? a.GetString() ?? "" : "row";
        if (Array.IndexOf(Arrangements, arrange) < 0)
        {
            throw bad($"{key}.arrange", $"is '{arrange}' (valid: {string.Join(", ", Arrangements)})");
        }

        var strokes = new List<DrawnStroke>();
        if (e.TryGetProperty("strokes", out var list))
        {
            foreach (var stroke in list.EnumerateArray())
            {
                strokes.Add(Stroke(stroke, key, bad));
            }
        }

        var place = e.TryGetProperty("place", out var p)
            ? p.GetString() switch { "above" => FontMarkPlace.Above, "below" => FontMarkPlace.Below, _ => throw bad($"{key}.place", "is not above or below") }
            : FontMarkPlace.Above;
        bool strip = e.TryGetProperty("strip", out var s) && (s.ValueKind is JsonValueKind.True or JsonValueKind.False ? s.GetBoolean() : throw bad($"{key}.strip", "is true or false"));
        bool bar = e.TryGetProperty("bar", out var b) && (b.ValueKind == JsonValueKind.String && b.GetString() == "top" ? true : throw bad($"{key}.bar", "is \"top\" (a bar parallel to the top stroke)"));
        return new ComposedForm(code, form, dots, arrange, [.. strokes], place) { Strip = strip, Bar = bar };
    }

    private static DrawnStroke Stroke(JsonElement stroke, string key, Func<string, string, InvalidDataException> bad)
    {
        static double[] Points(JsonElement points) => [.. points.EnumerateArray().SelectMany(p => p.EnumerateArray().Select(c => c.GetDouble()))];
        int Int(string name, int fallback) => stroke.TryGetProperty(name, out var e) && e.TryGetInt32(out int v) && v > 0 ? v : fallback;
        if (stroke.TryGetProperty("spline", out var spline))
        {
            var p = Points(spline);
            return p.Length >= 4 ? new DrawnStroke(DrawnStrokeKind.Spline, p, Int("per", 6), 0) : throw bad($"{key}.spline", "needs at least two points");
        }

        if (stroke.TryGetProperty("poly", out var poly))
        {
            var p = Points(poly);
            return p.Length >= 4 ? new DrawnStroke(DrawnStrokeKind.Poly, p, Int("per", 5), 0) : throw bad($"{key}.poly", "needs at least two points");
        }

        if (stroke.TryGetProperty("ell", out var ell))
        {
            double[] a = [.. ell.EnumerateArray().Select(e => e.GetDouble())];
            double shrink = stroke.TryGetProperty("shrink", out var sh) ? sh.GetDouble() : 0;
            return a.Length == 6 ? new DrawnStroke(DrawnStrokeKind.Ellipse, a, Int("samples", 30), shrink)
                : throw bad($"{key}.ell", "needs centre x, y, radii x, y and the start and end angles in degrees");
        }

        throw bad(key, "holds a stroke that is none of spline, poly or ell");
    }
}

/// <summary>How a drawn set is sized for one font: its box, pen and side bearing, in font units.</summary>
internal readonly record struct DrawnSize(double Width, double Height, double Pen, double SideBearing);

/// <summary>Draws the strokes of drawn glyphs into outlines: placed, wobbled by the seed, slanted, thickened into round-ended strokes of one winding.</summary>
internal static class StrokeDrawing
{
    /// <summary>
    /// The outline of each glyph of <paramref name="set"/>, in order, and each one's advance (0 for a mark): every glyph
    /// drawn from one generator seeded with <see cref="FontMapDrawing.Seed"/>, so a glyph does not change with which
    /// others a font lacks.
    /// </summary>
    public static (GlyphOutline Outline, int Advance)[] Draw(DrawnSet set, DrawnSize size, FontMapDrawing drawing, int discSides)
    {
        var random = new Random(drawing.Seed);
        var result = new (GlyphOutline, int)[set.Glyphs.Length];
        for (int i = 0; i < result.Length; i++)
        {
            var contours = new List<(double X, double Y)[]>();
            foreach (var stroke in set.Glyphs[i].Strokes)
            {
                var centre = stroke.Path();
                var path = new List<(double X, double Y)>(centre.Count);
                foreach (var (u, v) in centre)
                {
                    double du = Uniform(random, -drawing.Wobble, drawing.Wobble), dv = Uniform(random, -drawing.Wobble, drawing.Wobble);
                    double x = size.SideBearing + size.Pen / 2 + (u + du) * size.Width;
                    double y = size.Pen / 2 + (v + dv) * (set.Mark ? size.Height : size.Height - size.Pen);
                    path.Add((x + drawing.Slant * y, y));
                }

                Capsules(path, size.Pen * Uniform(random, drawing.PenScaleLow, drawing.PenScaleHigh), discSides, contours);
            }

            result[i] = ToOutline(contours, set.Mark, size.SideBearing);
        }

        return result;
    }

    private static double Uniform(Random random, double low, double high) => low + (high - low) * random.NextDouble();

    // Every segment a quad of the stroke's width and a disc at every other point and the last, all clockwise (y up), so
    // overlaps fill once under the nonzero rule.
    private static void Capsules(List<(double X, double Y)> path, double width, int sides, List<(double X, double Y)[]> contours)
    {
        double r = width / 2;
        for (int i = 0; i + 1 < path.Count; i++)
        {
            var (x0, y0) = path[i];
            var (x1, y1) = path[i + 1];
            double dx = x1 - x0, dy = y1 - y0, d = Math.Sqrt(dx * dx + dy * dy);
            if (d < 1e-6)
            {
                continue;
            }

            double nx = -dy / d * r, ny = dx / d * r;
            (double, double)[] quad = [(x0 + nx, y0 + ny), (x1 + nx, y1 + ny), (x1 - nx, y1 - ny), (x0 - nx, y0 - ny)];
            if (Area(quad) > 0)
            {
                Array.Reverse(quad);
            }

            contours.Add(quad);
        }

        for (int i = 0; i < path.Count; i++)
        {
            if (i % 2 != 0 && i != path.Count - 1)
            {
                continue;
            }

            var disc = new (double, double)[sides];
            for (int k = 0; k < sides; k++)
            {
                double a = -2 * Math.PI * k / sides;
                disc[k] = (path[i].X + r * PortableMath.Cos(a), path[i].Y + r * PortableMath.Sin(a));
            }

            contours.Add(disc);
        }
    }

    /// <summary>The signed area of a closed polygon (positive: counter-clockwise, y up).</summary>
    public static double Area(ReadOnlySpan<(double X, double Y)> c)
    {
        double sum = 0;
        for (int i = 0; i < c.Length; i++)
        {
            var (x0, y0) = c[i];
            var (x1, y1) = c[(i + 1) % c.Length];
            sum += x0 * y1 - x1 * y0;
        }

        return sum / 2;
    }

    // Points rounded to whole font units; a base glyph's advance is its ink's right edge plus the side bearing.
    private static (GlyphOutline, int) ToOutline(List<(double X, double Y)[]> contours, bool mark, double sideBearing)
    {
        int points = contours.Sum(c => c.Length);
        var verbs = new OutlineVerb[points + contours.Count];
        var xy = new float[2 * points];
        int v = 0, p = 0;
        float maxX = float.MinValue;
        foreach (var contour in contours)
        {
            for (int k = 0; k < contour.Length; k++)
            {
                verbs[v++] = k == 0 ? OutlineVerb.Move : OutlineVerb.Line;
                float x = (float)Math.Round(contour[k].X, MidpointRounding.ToEven), y = (float)Math.Round(contour[k].Y, MidpointRounding.ToEven);
                xy[p++] = x;
                xy[p++] = y;
                maxX = Math.Max(maxX, x);
            }

            verbs[v++] = OutlineVerb.Close;
        }

        var outline = points == 0 ? GlyphOutline.Empty : new GlyphOutline(verbs, xy);
        return (outline, mark || points == 0 ? 0 : (int)Math.Round(maxX + sideBearing, MidpointRounding.ToEven));
    }
}

/// <summary>
/// Letters made from a font's own glyphs: a glyph's contours split, its dots (small contours wound like its body)
/// replaced by copies of one of them arranged in a row or a triangle where they were, and strokes added on its ink.
/// </summary>
internal static class Composition
{
    // A dot is at most this share of the em on its larger side (a share of the glyph would miss a narrow form's dot).
    private const double DotShare = 0.18;

    /// <summary>The composed outline: <paramref name="body"/>'s contours, its dots as <paramref name="recipe"/> says, and <paramref name="strokes"/> (a mark's outline at its origin) placed on its ink.</summary>
    public static GlyphOutline Compose(GlyphOutline body, ComposedForm recipe, GlyphOutline? strokes, int unitsPerEm, int gap, int window)
    {
        var contours = Split(body);
        if (recipe.Strip && contours.Count > 1)
        {
            // The body's separate marks: contours wound as its main one, smaller, and inside its box (holes are wound the other way).
            var main = contours.MaxBy(c => Math.Abs(Area(c.Points)));
            int sign = Math.Sign(Area(main.Points));
            var (mx0, my0, mx1, my1) = Box(main.Points);
            contours = [.. contours.Where(c => c == main || Math.Sign(Area(c.Points)) != sign || !Inside(Box(c.Points), (mx0, my0, mx1, my1)))];
        }

        var result = new List<(OutlineVerb[] Verbs, float[] Points)>();
        if (recipe.Dots is int count && contours.Count > 1)
        {
            double side = unitsPerEm;
            var main = contours.MaxBy(c => Math.Abs(Area(c.Points)));
            double largest = Math.Abs(Area(main.Points));
            int sign = Math.Sign(Area(main.Points));
            var dots = contours.Where(c => Math.Sign(Area(c.Points)) == sign && Math.Abs(Area(c.Points)) < largest && Size(c.Points) <= DotShare * side).ToList();
            result.AddRange(contours.Where(c => !dots.Contains(c)));
            if (dots.Count > 0 && count > 0)
            {
                // Copies of the first dot, centred where the dots were: in a row, or a triangle with its point up or down.
                var template = dots[0];
                var (tx, ty) = Centre(template.Points);
                double cx = dots.Average(d => Centre(d.Points).X), cy = dots.Average(d => Centre(d.Points).Y);
                double step = 1.25 * Size(template.Points);
                foreach (var (dx, dy) in Places(count, recipe.Arrange, step))
                {
                    result.Add((template.Verbs, Moved(template.Points, cx + dx - tx, cy + dy - ty)));
                }
            }
        }
        else
        {
            result.AddRange(contours);
        }

        if (recipe.Bar && contours.Count > 0 && InkMeasure.Of(body, unitsPerEm, window) is { } stroked)
        {
            var main = contours.MaxBy(c => Math.Abs(Area(c.Points)));
            if (TopBar(main.Points, stroked.Thickness, Math.Sign(Area(main.Points))) is { } bar)
            {
                result.Add(bar);
            }
        }

        if (strokes is { IsEmpty: false } && InkMeasure.Of(body, unitsPerEm, window) is { } ink)
        {
            var (sx0, sy0, sx1, sy1) = strokes.Bounds();
            double dx = ink.CenterX - (sx0 + sx1) / 2;
            double dy = recipe.Place == FontMarkPlace.Below ? ink.MinY - gap - sy1 : ink.MaxY + gap - sy0;
            result.AddRange(Split(strokes).Select(c => (c.Verbs, Moved(c.Points, dx, dy))));
        }

        return result.Count == 0 ? GlyphOutline.Empty
            : new GlyphOutline([.. result.SelectMany(c => c.Verbs)], [.. result.SelectMany(c => c.Points)]);
    }

    private static IEnumerable<(double X, double Y)> Places(int count, string arrange, double step) => (count, arrange) switch
    {
        (1, _) => [(0, 0)],
        (2, _) => [(-step / 2, 0), (step / 2, 0)],
        (3, "up") => [(0, step * 0.43), (-step / 2, -step * 0.43), (step / 2, -step * 0.43)],
        (3, "down") => [(-step / 2, step * 0.43), (step / 2, step * 0.43), (0, -step * 0.43)],
        _ => Enumerable.Range(0, count).Select(i => ((i - (count - 1) / 2.0) * step, 0.0)),
    };

    /// <summary>The contours of an outline, each its verbs (from its move to its close) and their points.</summary>
    public static List<(OutlineVerb[] Verbs, float[] Points)> Split(GlyphOutline outline)
    {
        var verbs = outline.Verbs.Span;
        var points = outline.Points.Span;
        var result = new List<(OutlineVerb[], float[])>();
        int start = 0, p = 0, pointStart = 0;
        for (int v = 0; v < verbs.Length; v++)
        {
            if (verbs[v] == OutlineVerb.Move && v > start)
            {
                result.Add((verbs[start..v].ToArray(), points[pointStart..p].ToArray()));
                start = v;
                pointStart = p;
            }

            p += verbs[v] switch { OutlineVerb.Move or OutlineVerb.Line => 2, OutlineVerb.Quadratic => 4, OutlineVerb.Cubic => 6, _ => 0 };
        }

        if (verbs.Length > start)
        {
            result.Add((verbs[start..].ToArray(), points[pointStart..p].ToArray()));
        }

        return result;
    }

    // A bar parallel to a contour's top stroke, above it: the stroke's direction the main axis of the contour's points in
    // its top band (a near-upright stroke takes a gentle rising slant instead), its length theirs along it, its pen the
    // glyph's; round-ended, wound as the contour.
    private static (OutlineVerb[] Verbs, float[] Points)? TopBar(float[] contour, double pen, int sign)
    {
        var (x0, y0, x1, y1) = Box(contour);
        double band = y1 - Math.Max((y1 - y0) * 0.22, pen * 1.5);
        double sx = 0, sy = 0;
        int n = 0;
        for (int i = 0; i < contour.Length; i += 2)
        {
            if (contour[i + 1] >= band)
            {
                (sx, sy, n) = (sx + contour[i], sy + contour[i + 1], n + 1);
            }
        }

        if (n < 2 || !(pen > 0))
        {
            return null;
        }

        double mx = sx / n, my = sy / n, cxx = 0, cyy = 0, cxy = 0;
        for (int i = 0; i < contour.Length; i += 2)
        {
            if (contour[i + 1] >= band)
            {
                double dx0 = contour[i] - mx, dy0 = contour[i + 1] - my;
                (cxx, cyy, cxy) = (cxx + dx0 * dx0, cyy + dy0 * dy0, cxy + dx0 * dy0);
            }
        }

        double angle = 0.5 * PortableMath.Atan2(2 * cxy, cxx - cyy);
        double ux = PortableMath.Cos(angle), uy = PortableMath.Sin(angle);
        if (ux < 0)
        {
            (ux, uy) = (-ux, -uy);
        }

        double length;
        if (Math.Abs(uy) > 0.85)
        {
            // An upright top (a stem): a short bar rising gently, as a hand adds one.
            (ux, uy) = (1 / Math.Sqrt(1 + 0.35 * 0.35), 0.35 / Math.Sqrt(1 + 0.35 * 0.35));
            length = Math.Max(4 * pen, 0.3 * (y1 - y0));
        }
        else
        {
            double low = double.MaxValue, high = double.MinValue;
            for (int i = 0; i < contour.Length; i += 2)
            {
                if (contour[i + 1] >= band)
                {
                    double along = (contour[i] - mx) * ux + (contour[i + 1] - my) * uy;
                    (low, high) = (Math.Min(low, along), Math.Max(high, along));
                }
            }

            length = Math.Max(2 * pen, high - low);
        }

        // Above the stroke's middle line by its half, a gap and the bar's half: about two pens.
        double nx = -uy, ny = ux;
        if (ny < 0)
        {
            (nx, ny) = (-nx, -ny);
        }

        double cx = mx + nx * 2.2 * pen, cy = my + ny * 2.2 * pen, half = Math.Max(0, length / 2 - pen / 2), r = pen / 2;
        const int Cap = 8;
        var points = new List<float>(4 * (Cap + 1));
        void Add(double x, double y)
        {
            points.Add((float)Math.Round(x, MidpointRounding.ToEven));
            points.Add((float)Math.Round(y, MidpointRounding.ToEven));
        }

        // One side, the round end at the far end, the other side, the round end at the near end.
        for (int end = 0; end < 2; end++)
        {
            double ex = cx + (end == 0 ? half : -half) * ux, ey = cy + (end == 0 ? half : -half) * uy;
            double start = PortableMath.Atan2(end == 0 ? -ux : ux, end == 0 ? uy : -uy);
            for (int k = 0; k <= Cap; k++)
            {
                double a = start + Math.PI * k / Cap;
                Add(ex + r * PortableMath.Cos(a), ey + r * PortableMath.Sin(a));
            }
        }

        var p = points.ToArray();
        if (Math.Sign(Area(p)) != sign)
        {
            var reversed = new float[p.Length];
            for (int i = 0; i < p.Length; i += 2)
            {
                reversed[p.Length - 2 - i] = p[i];
                reversed[p.Length - 1 - i] = p[i + 1];
            }

            p = reversed;
        }

        var verbs = new OutlineVerb[p.Length / 2 + 1];
        verbs[0] = OutlineVerb.Move;
        for (int i = 1; i < p.Length / 2; i++)
        {
            verbs[i] = OutlineVerb.Line;
        }

        verbs[^1] = OutlineVerb.Close;
        return (verbs, p);
    }

    private static (float X0, float Y0, float X1, float Y1) Box(float[] p)
    {
        float minX = float.MaxValue, maxX = float.MinValue, minY = float.MaxValue, maxY = float.MinValue;
        for (int i = 0; i < p.Length; i += 2)
        {
            minX = Math.Min(minX, p[i]);
            maxX = Math.Max(maxX, p[i]);
            minY = Math.Min(minY, p[i + 1]);
            maxY = Math.Max(maxY, p[i + 1]);
        }

        return (minX, minY, maxX, maxY);
    }

    private static bool Inside((float X0, float Y0, float X1, float Y1) inner, (float X0, float Y0, float X1, float Y1) outer) =>
        inner.X0 >= outer.X0 && inner.Y0 >= outer.Y0 && inner.X1 <= outer.X1 && inner.Y1 <= outer.Y1;

    private static double Area(float[] p)
    {
        double sum = 0;
        int n = p.Length / 2;
        for (int i = 0; i < n; i++)
        {
            int j = (i + 1) % n;
            sum += (double)p[2 * i] * p[2 * j + 1] - (double)p[2 * j] * p[2 * i + 1];
        }

        return sum / 2;
    }

    private static double Size(float[] p)
    {
        float minX = float.MaxValue, maxX = float.MinValue, minY = float.MaxValue, maxY = float.MinValue;
        for (int i = 0; i < p.Length; i += 2)
        {
            minX = Math.Min(minX, p[i]);
            maxX = Math.Max(maxX, p[i]);
            minY = Math.Min(minY, p[i + 1]);
            maxY = Math.Max(maxY, p[i + 1]);
        }

        return Math.Max(maxX - minX, maxY - minY);
    }

    private static (double X, double Y) Centre(float[] p)
    {
        float minX = float.MaxValue, maxX = float.MinValue, minY = float.MaxValue, maxY = float.MinValue;
        for (int i = 0; i < p.Length; i += 2)
        {
            minX = Math.Min(minX, p[i]);
            maxX = Math.Max(maxX, p[i]);
            minY = Math.Min(minY, p[i + 1]);
            maxY = Math.Max(maxY, p[i + 1]);
        }

        return ((minX + maxX) / 2.0, (minY + maxY) / 2.0);
    }

    private static float[] Moved(float[] p, double dx, double dy)
    {
        var moved = new float[p.Length];
        for (int i = 0; i < p.Length; i += 2)
        {
            moved[i] = (float)Math.Round(p[i] + dx, MidpointRounding.ToEven);
            moved[i + 1] = (float)Math.Round(p[i + 1] + dy, MidpointRounding.ToEven);
        }

        return moved;
    }
}
