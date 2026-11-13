// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Frozen;

namespace Idrak.Imaging;

/// <summary>What the built-in shapes share: their paths in the step's width, dash and tone, arcs, regions inset from boxes.</summary>
internal static class ShapeGeometry
{
    /// <summary>The padding inside a region, in pixels: the step's <c>pad</c> (multiples of the text size).</summary>
    public static float Pad(ShapeStep step, ShapeContext context, float otherwise) => step.Number("pad", otherwise, 0, 20) * context.TextSize;

    /// <summary>Half a path's stroke in pixels, for a path of <paramref name="factor"/> times the step's width.</summary>
    public static float Half(ShapeStep step, ShapeContext context, float factor = 1) => context.StrokeWidth * step.WidthFactor() * factor / 2;

    /// <summary>A path in the step's width (times <paramref name="factor"/>), dash and tone.</summary>
    public static ShapePath Path(ShapeStep step, float[] points, bool closed, float factor = 1) => new(points, closed, step.WidthFactor() * factor, step.Dash(), step.Tone());

    /// <summary>Refuses a value on a step whose shape takes none.</summary>
    public static void NoValue(ShapeStep step)
    {
        if (!string.IsNullOrEmpty(step.Value))
        {
            throw new ArgumentException($"{step.Name} takes no value (given '{step.Value}').");
        }
    }

    /// <summary>
    /// An elliptical arc from <paramref name="from"/> to <paramref name="to"/> degrees (0 toward the end side, the reading
    /// direction; counterclockwise as seen), flattened within the context's flatness; a closed ring when
    /// <paramref name="closed"/> (its last point not repeated).
    /// </summary>
    public static float[] Arc(ShapeContext context, float cx, float cy, float rx, float ry, double from, double to, bool closed)
    {
        double sweep = (to - from) * Math.PI / 180;
        float r = Math.Max(rx, ry);
        // A chord of angle θ strays r θ² / 8 from its arc.
        int segments = r <= 0 ? 1 : Math.Clamp((int)Math.Ceiling(Math.Abs(sweep) * Math.Sqrt(r / (8 * Math.Max(1e-3f, context.Flatness)))), 4, 4096);
        int count = closed ? segments : segments + 1;
        var points = new float[2 * count];
        float direction = context.RightToLeft ? -1 : 1;
        for (int k = 0; k < count; k++)
        {
            double a = from * Math.PI / 180 + sweep * k / segments;
            points[2 * k] = cx + direction * rx * (float)PortableMath.Cos(a);
            points[2 * k + 1] = cy - ry * (float)PortableMath.Sin(a);
        }

        return points;
    }

    /// <summary>The region between the box's edges, each inset by its own amount (an empty region at the middle when nothing is left).</summary>
    public static ShapeRegion Region(int group, int index, float x0, float y0, float x1, float y1, float left, float top, float right, float bottom)
    {
        float x = x0 + left, y = y0 + top, w = x1 - right - x, h = y1 - bottom - y;
        if (w < 0)
        {
            (x, w) = ((x0 + x1) / 2, 0);
        }

        if (h < 0)
        {
            (y, h) = ((y0 + y1) / 2, 0);
        }

        return new ShapeRegion(group, index, x, y, w, h);
    }

    /// <summary>The point at a share pair of the context's box (x from the start side).</summary>
    public static (float X, float Y) At(ShapeContext context, (float A, float B) share) => (context.XAt(share.A), context.YAt(share.B));

    /// <summary>How far along (x0, y0) → (x1, y1) the segment stays inside the context's box, from 0 to 1 (Liang–Barsky).</summary>
    public static (float From, float To) Inside(ShapeContext context, float x0, float y0, float x1, float y1)
    {
        float from = 0, to = 1, dx = x1 - x0, dy = y1 - y0;
        foreach (var (p, q) in ((float, float)[])[(-dx, x0 - context.X), (dx, context.Right - x0), (-dy, y0 - context.Y), (dy, context.Bottom - y0)])
        {
            if (p == 0)
            {
                if (q < 0)
                {
                    return (1, 0);
                }

                continue;
            }

            float r = q / p;
            if (p < 0)
            {
                from = Math.Max(from, r);
            }
            else
            {
                to = Math.Min(to, r);
            }
        }

        return (from, to);
    }
}

/// <summary>"line": a straight line between two points of its box, or from a point at a length and an angle.</summary>
internal sealed class LineShape : IShape
{
    private static readonly OptionKeys Own = new("from", "to", "length", "angle");

    public string Name => "line";

    public string Summary => "a straight line: from=X:Y to=X:Y (shares of its box, X from the start side; 0:0.5 to 1:0.5), or from, length (a share of the box's width) and angle (degrees from the reading direction, up positive)";

    public IReadOnlySet<string> Keys => Own;

    public void Check(ShapeStep step)
    {
        ShapeGeometry.NoValue(step);
        _ = step.Pair("from", (0, 0.5f), 0, 1);
        _ = step.Pair("to", (1, 0.5f), 0, 1);
        _ = step.Number("length", 1, 0, 1);
        _ = step.Number("angle", 0, -360, 360);
        if (step.Has("to") && (step.Has("length") || step.Has("angle")))
        {
            throw new ArgumentException("A line goes to a point (to) or at a length and an angle (length, angle), not both.");
        }
    }

    public ShapePlan Plan(ShapeStep step, ShapeContext context)
    {
        var (x0, y0) = ShapeGeometry.At(context, step.Pair("from", (0, 0.5f), 0, 1));
        float x1, y1;
        if (step.Has("length") || step.Has("angle"))
        {
            float length = step.Number("length", 1, 0, 1) * context.Width;
            double angle = step.Number("angle", 0, -360, 360) * Math.PI / 180;
            x1 = x0 + (context.RightToLeft ? -1 : 1) * length * (float)PortableMath.Cos(angle);
            y1 = y0 - length * (float)PortableMath.Sin(angle);
            var (_, to) = ShapeGeometry.Inside(context, x0, y0, x1, y1);   // kept inside its box
            (x1, y1) = (x0 + (x1 - x0) * Math.Max(0, to), y0 + (y1 - y0) * Math.Max(0, to));
        }
        else
        {
            (x1, y1) = ShapeGeometry.At(context, step.Pair("to", (1, 0.5f), 0, 1));
        }

        return new ShapePlan([ShapeGeometry.Path(step, [x0, y0, x1, y1], closed: false)]);
    }
}

/// <summary>"polyline": an open line through points of its box.</summary>
internal sealed class PolylineShape : IShape
{
    private static readonly OptionKeys Own = new("points");

    public string Name => "polyline";

    public string Summary => "an open line through points=X:Y:X:Y:... (shares of its box, X from the start side; at least two)";

    public IReadOnlySet<string> Keys => Own;

    public void Check(ShapeStep step)
    {
        ShapeGeometry.NoValue(step);
        _ = step.Points("points", 2) ?? throw new ArgumentException("A polyline needs points=X:Y:X:Y:... (at least two).");
    }

    public ShapePlan Plan(ShapeStep step, ShapeContext context)
    {
        var shares = step.Points("points", 2) ?? [0, 0.5f, 1, 0.5f];
        var points = new float[shares.Length];
        for (int i = 0; i < shares.Length; i += 2)
        {
            (points[i], points[i + 1]) = ShapeGeometry.At(context, (shares[i], shares[i + 1]));
        }

        return new ShapePlan([ShapeGeometry.Path(step, points, closed: false)]);
    }
}

/// <summary>"polygon": a regular polygon of N sides in its box (one region inside), or a closed line through points.</summary>
internal sealed class PolygonShape : IShape
{
    private static readonly OptionKeys Own = new("points", "pad");

    // The value ("") is drawn whole by a profile: a count of sides.
    private static readonly FrozenSet<string> Whole = new[] { "" }.ToFrozenSet(StringComparer.Ordinal);

    public string Name => "polygon";

    public string Summary => "a closed polygon: regular with N sides (polygon=N; 6) in its box, a region inside it, or through points=X:Y:X:Y:... (shares, X from the start side); pad (text sizes; 0.15)";

    public IReadOnlySet<string> Keys => Own;

    public IReadOnlySet<string> WholeKeys => Whole;

    public void Check(ShapeStep step)
    {
        _ = step.Integer("", 6, 3, 512);
        _ = step.Points("points", 3);
        _ = step.Number("pad", 0.15f, 0, 20);
        if (step.Has("") && step.Has("points"))
        {
            throw new ArgumentException("A polygon has N sides (polygon=N) or points, not both.");
        }
    }

    public ShapePlan Plan(ShapeStep step, ShapeContext context)
    {
        if (step.Points("points", 3) is { } shares)
        {
            var through = new float[shares.Length];
            for (int i = 0; i < shares.Length; i += 2)
            {
                (through[i], through[i + 1]) = ShapeGeometry.At(context, (shares[i], shares[i + 1]));
            }

            return new ShapePlan([ShapeGeometry.Path(step, through, closed: true)]);
        }

        int sides = step.Integer("", 6, 3, 512);
        float cx = context.X + context.Width / 2, cy = context.Y + context.Height / 2, rx = context.Width / 2, ry = context.Height / 2;
        var points = new float[2 * sides];
        for (int k = 0; k < sides; k++)
        {
            double a = Math.PI / 2 + 2 * Math.PI * k / sides;               // the first corner at the top
            points[2 * k] = cx + (context.RightToLeft ? -1 : 1) * rx * (float)PortableMath.Cos(a);
            points[2 * k + 1] = cy - ry * (float)PortableMath.Sin(a);
        }

        // The square inside the polygon's inner circle, scaled to its box, less the stroke and the padding.
        float inner = (float)PortableMath.Cos(Math.PI / sides) / MathF.Sqrt(2);
        float inset = ShapeGeometry.Half(step, context) + ShapeGeometry.Pad(step, context, 0.15f);
        var region = ShapeGeometry.Region(0, 0, cx - rx * inner, cy - ry * inner, cx + rx * inner, cy + ry * inner, inset, inset, inset, inset);
        return new ShapePlan([ShapeGeometry.Path(step, points, closed: true)], null, [region]);
    }
}

/// <summary>"box": a rectangle around its box (rounded corners, some sides only), one region inside.</summary>
internal sealed class BoxShape : IShape
{
    private static readonly OptionKeys Own = new("radius", "sides", "pad");

    private static readonly string[] SideNames = ["top", "end", "bottom", "start"];

    public string Name => "box";

    public string Summary => "a rectangle around its box, a region inside: radius (rounded corners, a share of the smaller side; 0), sides=top:end:bottom:start (the sides drawn; all), pad (text sizes; 0.15)";

    public IReadOnlySet<string> Keys => Own;

    public void Check(ShapeStep step)
    {
        ShapeGeometry.NoValue(step);
        _ = step.Number("radius", 0, 0, 0.5f);
        _ = Sides(step);
        _ = step.Number("pad", 0.15f, 0, 20);
    }

    public ShapePlan Plan(ShapeStep step, ShapeContext context)
    {
        var sides = Sides(step);
        float r = step.Number("radius", 0, 0, 0.5f) * Math.Min(context.Width, context.Height);
        float start = context.RightToLeft ? context.Right : context.X, end = context.RightToLeft ? context.X : context.Right;
        // The corners in drawing order, each side from one to the next: top, end, bottom, start.
        (float X, float Y)[] corners = [(start, context.Y), (end, context.Y), (end, context.Bottom), (start, context.Bottom)];
        var paths = new List<ShapePath>();
        if (sides.All(s => s))
        {
            paths.Add(ShapeGeometry.Path(step, Outline(context, corners, r, 0, 4, closed: true), closed: true));
        }
        else
        {
            // Each run of sides drawn, as one path: around the corners between them, from the corner before it to the one after.
            int first = Array.IndexOf(sides, false) + 1;
            for (int k = 0; k < 4;)
            {
                int i = (first + k) % 4;
                if (!sides[i])
                {
                    k++;
                    continue;
                }

                int run = 0;
                while (k + run < 4 && sides[(first + k + run) % 4])
                {
                    run++;
                }

                paths.Add(ShapeGeometry.Path(step, Outline(context, corners, r, i, run, closed: false), closed: false));
                k += run;
            }
        }

        float inset = ShapeGeometry.Half(step, context) + ShapeGeometry.Pad(step, context, 0.15f) + 0.3f * r;
        var region = ShapeGeometry.Region(0, 0, context.X, context.Y, context.Right, context.Bottom, inset, inset, inset, inset);
        return new ShapePlan(paths, null, [region]);
    }

    // The sides drawn, in drawing order (top, end, bottom, start).
    private static bool[] Sides(ShapeStep step)
    {
        var drawn = new bool[4];
        if (step.List("sides") is not { } names)
        {
            Array.Fill(drawn, true);
            return drawn;
        }

        foreach (string name in names)
        {
            drawn[Array.IndexOf(SideNames, PageOptions.Choice("box's sides", name, SideNames))] = true;
        }

        return drawn;
    }

    // The path along `count` sides from side `from` (corner `from` to corner `from + count`), corners between them
    // rounded by `r`; a closed one rounds every corner.
    private static float[] Outline(ShapeContext context, (float X, float Y)[] corners, float r, int from, int count, bool closed)
    {
        var points = new List<float>();
        void Add(float x, float y)
        {
            points.Add(x);
            points.Add(y);
        }

        for (int k = 0; k <= count; k++)
        {
            int i = (from + k) % 4;
            var corner = corners[i];
            bool inner = closed || (k > 0 && k < count);
            if (!inner || r <= 0)
            {
                if (!closed || k < count)
                {
                    Add(corner.X, corner.Y);
                }

                continue;
            }

            if (closed && k == count)
            {
                break;
            }

            // The arc at this corner from the side coming in (from corner i - 1) to the side going out (to corner i + 1).
            var (px, py) = corners[(i + 3) % 4];
            var (nx, ny) = corners[(i + 1) % 4];
            float inX = Math.Sign(corner.X - px), inY = Math.Sign(corner.Y - py), outX = Math.Sign(nx - corner.X), outY = Math.Sign(ny - corner.Y);
            float qx = corner.X - r * inX + r * outX, qy = corner.Y - r * inY + r * outY;
            int segments = Math.Clamp((int)Math.Ceiling(Math.PI / 2 * Math.Sqrt(r / (8 * Math.Max(1e-3f, context.Flatness)))), 2, 512);
            for (int s = 0; s <= segments; s++)
            {
                double t = Math.PI / 2 * s / segments;
                float c = (float)PortableMath.Cos(t), sn = (float)PortableMath.Sin(t);
                Add(qx + r * (-outX * c + inX * sn), qy + r * (-outY * c + inY * sn));
            }
        }

        return [.. points];
    }
}

/// <summary>
/// "grid": rows and columns, each line drawn once across the whole grid (shared edges are not doubled), a region a cell,
/// a group a row, columns in reading order from the start side.
/// </summary>
internal sealed class GridShape : IShape
{
    private static readonly OptionKeys Own = new("widths", "heights", "row-height", "heavy-every", "heavy-from", "heavy-width", "border", "inner", "pad");

    // The value ("": rows and columns) and the heavy lines' columns are drawn whole by a profile.
    private static readonly FrozenSet<string> Whole = new[] { "", "heavy-every", "heavy-from" }.ToFrozenSet(StringComparer.Ordinal);

    private static readonly string[] SideNames = ["top", "end", "bottom", "start"];

    public string Name => "grid";

    public string Summary => "a grid, grid=ROWSxCOLS (auto: a row a text line, a column a unit of its longest line; auto x auto), each line drawn once across it, a region a cell: "
        + "widths and heights (relative, columns in reading order), row-height (text line heights; 1.3..2), heavy-every, heavy-from and heavy-width (heavier lines after column "
        + "heavy-from and every heavy-every after it, heavy-width times as wide; 2), border (on, off or sides top:end:bottom:start), inner (on, off, rows, cols), pad (text sizes; 0.12)";

    public IReadOnlySet<string> Keys => Own;

    public IReadOnlySet<string> WholeKeys => Whole;

    public void Check(ShapeStep step)
    {
        _ = Size(step, new ShapeContext());
        _ = step.Numbers("widths", 1e-4f, 1e6f);
        _ = step.Numbers("heights", 1e-4f, 1e6f);
        _ = step.Range("row-height", (1.3f, 2f), 0.2f, 50);
        _ = step.Integer("heavy-every", 0, 0, 100_000);
        _ = step.Integer("heavy-from", 0, 0, 100_000);
        _ = step.Number("heavy-width", 2, 0.1f, 50);
        _ = Border(step);
        _ = step.Choice("inner", "on", "on", "off", "rows", "cols");
        _ = step.Number("pad", 0.12f, 0, 20);
    }

    public ShapePlan Plan(ShapeStep step, ShapeContext context)
    {
        var (rows, cols) = Size(step, context);
        var widths = step.Numbers("widths", 1e-4f, 1e6f);
        var heights = step.Numbers("heights", 1e-4f, 1e6f);
        int every = step.Integer("heavy-every", 0, 0, 100_000), heavyFrom = step.Integer("heavy-from", 0, 0, 100_000);
        float heavy = step.Number("heavy-width", 2, 0.1f, 50);
        var border = Border(step);
        string inner = step.Choice("inner", "on", "on", "off", "rows", "cols");
        float pad = ShapeGeometry.Pad(step, context, 0.12f);

        // Column edges from the start side, row edges from the top.
        var xs = new float[cols + 1];
        double weightSum = 0;
        for (int c = 0; c < cols; c++)
        {
            weightSum += Weight(widths, c);
        }

        double along = 0;
        float direction = context.RightToLeft ? -1 : 1, start = context.RightToLeft ? context.Right : context.X;
        for (int c = 0; c <= cols; c++)
        {
            xs[c] = c == cols ? start + direction * context.Width : start + direction * (float)(context.Width * along / weightSum);
            along += c < cols ? Weight(widths, c) : 0;
        }

        var ys = new float[rows + 1];
        ys[0] = context.Y;
        if (heights is not null)
        {
            double sum = 0, down = 0;
            for (int r = 0; r < rows; r++)
            {
                sum += Weight(heights, r);
            }

            for (int r = 0; r < rows; r++)
            {
                down += Weight(heights, r);
                ys[r + 1] = r == rows - 1 ? context.Bottom : context.Y + (float)(context.Height * down / sum);
            }
        }
        else
        {
            var range = step.Range("row-height", (1.3f, 2f), 0.2f, 50);
            float rowHeight = (range.Min + (float)new Random(context.Seed).NextDouble() * (range.Max - range.Min)) * context.LineHeight;
            if (!context.GrowsDown)
            {
                rowHeight = Math.Min(rowHeight, context.Height / rows);
            }

            for (int r = 0; r < rows; r++)
            {
                ys[r + 1] = ys[r] + rowHeight;
            }
        }

        // The width of each column line (0: not drawn) and each row line, as factors of the step's width.
        bool rowLines = inner is "on" or "rows", colLines = inner is "on" or "cols";
        var colWidth = new float[cols + 1];
        var rowWidth = new float[rows + 1];
        for (int c = 1; c < cols; c++)
        {
            bool heavier = every > 0 ? c >= heavyFrom && (c - heavyFrom) % every == 0 : heavyFrom > 0 && c == heavyFrom;
            colWidth[c] = colLines ? (heavier ? heavy : 1) : 0;
        }

        for (int r = 1; r < rows; r++)
        {
            rowWidth[r] = rowLines ? 1 : 0;
        }

        (rowWidth[0], colWidth[cols], rowWidth[rows], colWidth[0]) = (border[0] ? 1 : 0, border[1] ? 1 : 0, border[2] ? 1 : 0, border[3] ? 1 : 0);

        var paths = new List<ShapePath>();
        float left = Math.Min(xs[0], xs[cols]), right = Math.Max(xs[0], xs[cols]);
        if (border.All(b => b))
        {
            paths.Add(ShapeGeometry.Path(step, [xs[0], ys[0], xs[cols], ys[0], xs[cols], ys[rows], xs[0], ys[rows]], closed: true));
        }
        else
        {
            for (int r = 0; r <= rows; r += rows)
            {
                if (rowWidth[r] > 0)
                {
                    paths.Add(ShapeGeometry.Path(step, [xs[0], ys[r], xs[cols], ys[r]], closed: false));
                }
            }

            for (int c = 0; c <= cols; c += cols)
            {
                if (colWidth[c] > 0)
                {
                    paths.Add(ShapeGeometry.Path(step, [xs[c], ys[0], xs[c], ys[rows]], closed: false));
                }
            }
        }

        for (int r = 1; r < rows; r++)
        {
            if (rowWidth[r] > 0)
            {
                paths.Add(ShapeGeometry.Path(step, [left, ys[r], right, ys[r]], closed: false));
            }
        }

        for (int c = 1; c < cols; c++)
        {
            if (colWidth[c] > 0)
            {
                paths.Add(ShapeGeometry.Path(step, [xs[c], ys[0], xs[c], ys[rows]], closed: false, colWidth[c]));
            }
        }

        // A cell's region: inside the half widths of the lines around it, and the padding.
        float half = ShapeGeometry.Half(step, context);
        var regions = new List<ShapeRegion>(rows * cols);
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                float a = xs[c], b = xs[c + 1];
                float startInset = half * colWidth[c] + pad, endInset = half * colWidth[c + 1] + pad;
                regions.Add(ShapeGeometry.Region(r, c, Math.Min(a, b), ys[r], Math.Max(a, b), ys[r + 1],
                    context.RightToLeft ? endInset : startInset, half * rowWidth[r] + pad, context.RightToLeft ? startInset : endInset, half * rowWidth[r + 1] + pad));
            }
        }

        return new ShapePlan(paths, null, regions);
    }

    private static double Weight(float[]? weights, int index) => weights is not null && index < weights.Length ? weights[index] : 1;

    // ROWSxCOLS, each a number or auto (auto x auto when not given).
    private static (int Rows, int Cols) Size(ShapeStep step, ShapeContext context)
    {
        string text = step.Option("", "auto x auto")!;
        int x = text.IndexOfAny(['x', 'X', '×']);
        if (x < 0)
        {
            throw new ArgumentException($"grid's value is ROWSxCOLS, each a number or auto (such as 3x4 or auto x 18), not '{text}'.");
        }

        int Count(string part, int auto)
        {
            part = part.Trim();
            return part.Equals("auto", StringComparison.OrdinalIgnoreCase) ? Math.Max(1, auto) : PageOptions.Integer("grid's rows and columns", part, 1, 4096);
        }

        return (Count(text[..x], context.Lines), Count(text[(x + 1)..], context.MostUnits));
    }

    // The border's sides drawn, in drawing order (top, end, bottom, start): on (all), off (none) or a list of sides.
    private static bool[] Border(ShapeStep step)
    {
        var sides = new bool[4];
        string text = step.Option("border", "on")!;
        if (text.Contains(':') || Array.Exists(SideNames, s => s.Equals(text.Trim(), StringComparison.OrdinalIgnoreCase)))
        {
            foreach (string name in PageOptions.List("grid's border", text))
            {
                sides[Array.IndexOf(SideNames, PageOptions.Choice("grid's border", name, SideNames))] = true;
            }

            return sides;
        }

        Array.Fill(sides, PageOptions.Switch("grid's border", text));
        return sides;
    }
}

/// <summary>"circle": a ring in the middle of its box, one region inside.</summary>
internal sealed class CircleShape : IShape
{
    private static readonly OptionKeys Own = new("radius", "pad");

    public string Name => "circle";

    public string Summary => "a ring in the middle of its box, a region inside: radius (a share of half the smaller side; 1), pad (text sizes; 0.15)";

    public IReadOnlySet<string> Keys => Own;

    public void Check(ShapeStep step)
    {
        ShapeGeometry.NoValue(step);
        _ = step.Number("radius", 1, 0.01f, 1);
        _ = step.Number("pad", 0.15f, 0, 20);
    }

    public ShapePlan Plan(ShapeStep step, ShapeContext context)
    {
        float r = step.Number("radius", 1, 0.01f, 1) * Math.Min(context.Width, context.Height) / 2;
        float cx = context.X + context.Width / 2, cy = context.Y + context.Height / 2;
        var ring = ShapeGeometry.Arc(context, cx, cy, r, r, 0, 360, closed: true);
        float side = r / MathF.Sqrt(2), inset = ShapeGeometry.Half(step, context) + ShapeGeometry.Pad(step, context, 0.15f);
        return new ShapePlan([ShapeGeometry.Path(step, ring, closed: true)], null, [ShapeGeometry.Region(0, 0, cx - side, cy - side, cx + side, cy + side, inset, inset, inset, inset)]);
    }
}

/// <summary>"ellipse": an ellipse in its box, one region inside.</summary>
internal sealed class EllipseShape : IShape
{
    private static readonly OptionKeys Own = new("radii", "pad");

    public string Name => "ellipse";

    public string Summary => "an ellipse in its box, a region inside: radii=RX:RY (shares of half the width and half the height; 1:1), pad (text sizes; 0.15)";

    public IReadOnlySet<string> Keys => Own;

    public void Check(ShapeStep step)
    {
        ShapeGeometry.NoValue(step);
        _ = step.Pair("radii", (1, 1), 0.01f, 1);
        _ = step.Number("pad", 0.15f, 0, 20);
    }

    public ShapePlan Plan(ShapeStep step, ShapeContext context)
    {
        var (a, b) = step.Pair("radii", (1, 1), 0.01f, 1);
        float rx = a * context.Width / 2, ry = b * context.Height / 2, cx = context.X + context.Width / 2, cy = context.Y + context.Height / 2;
        var ring = ShapeGeometry.Arc(context, cx, cy, rx, ry, 0, 360, closed: true);
        float sx = rx / MathF.Sqrt(2), sy = ry / MathF.Sqrt(2), inset = ShapeGeometry.Half(step, context) + ShapeGeometry.Pad(step, context, 0.15f);
        return new ShapePlan([ShapeGeometry.Path(step, ring, closed: true)], null, [ShapeGeometry.Region(0, 0, cx - sx, cy - sy, cx + sx, cy + sy, inset, inset, inset, inset)]);
    }
}

/// <summary>"arc": part of an ellipse in its box.</summary>
internal sealed class ArcShape : IShape
{
    private static readonly OptionKeys Own = new("radii", "start", "end");

    public string Name => "arc";

    public string Summary => "part of an ellipse in its box: start and end (degrees from the reading direction, up positive; 0 and 180), radii=RX:RY (shares of half the width and height; 1:1)";

    public IReadOnlySet<string> Keys => Own;

    public void Check(ShapeStep step)
    {
        ShapeGeometry.NoValue(step);
        _ = step.Pair("radii", (1, 1), 0.01f, 1);
        float from = step.Number("start", 0, -720, 720), to = step.Number("end", 180, -720, 720);
        if (from == to || Math.Abs(to - from) > 360)
        {
            throw new ArgumentException($"An arc's start and end differ, by 360 degrees at most (start {from}, end {to}).");
        }
    }

    public ShapePlan Plan(ShapeStep step, ShapeContext context)
    {
        var (a, b) = step.Pair("radii", (1, 1), 0.01f, 1);
        float rx = a * context.Width / 2, ry = b * context.Height / 2, cx = context.X + context.Width / 2, cy = context.Y + context.Height / 2;
        var points = ShapeGeometry.Arc(context, cx, cy, rx, ry, step.Number("start", 0, -720, 720), step.Number("end", 180, -720, 720), closed: false);
        return new ShapePlan([ShapeGeometry.Path(step, points, closed: false)]);
    }
}

/// <summary>"arrow": a line with a head at its end.</summary>
internal sealed class ArrowShape : IShape
{
    private static readonly OptionKeys Own = new("from", "to", "head", "head-style");

    public string Name => "arrow";

    public string Summary => "a line with a head at its end: from=X:Y to=X:Y (shares of its box, X from the start side; 0:0.5 to 1:0.5), head (its length in stroke widths; 8), head-style (open, closed, filled)";

    public IReadOnlySet<string> Keys => Own;

    public void Check(ShapeStep step)
    {
        ShapeGeometry.NoValue(step);
        _ = step.Pair("from", (0, 0.5f), 0, 1);
        _ = step.Pair("to", (1, 0.5f), 0, 1);
        _ = step.Number("head", 8, 0, 200);
        _ = step.Choice("head-style", "open", "open", "closed", "filled");
    }

    public ShapePlan Plan(ShapeStep step, ShapeContext context)
    {
        var (x0, y0) = ShapeGeometry.At(context, step.Pair("from", (0, 0.5f), 0, 1));
        var (x1, y1) = ShapeGeometry.At(context, step.Pair("to", (1, 0.5f), 0, 1));
        var paths = new List<ShapePath> { ShapeGeometry.Path(step, [x0, y0, x1, y1], closed: false) };
        var areas = new List<ShapeArea>();
        float dx = x1 - x0, dy = y1 - y0, length = MathF.Sqrt(dx * dx + dy * dy);
        float head = Math.Min(step.Number("head", 8, 0, 200) * context.StrokeWidth * step.WidthFactor(), length * 0.6f);
        if (length > 0 && head > 0)
        {
            // The barbs 25 degrees off the line, back from its tip, kept inside the box.
            const float Cos = 0.9063078f, Sin = 0.42261827f;
            float ux = dx / length, uy = dy / length;
            float ax = x1 - head * (ux * Cos - uy * Sin), ay = y1 - head * (uy * Cos + ux * Sin);
            float bx = x1 - head * (ux * Cos + uy * Sin), by = y1 - head * (uy * Cos - ux * Sin);
            (ax, ay, bx, by) = (Math.Clamp(ax, context.X, context.Right), Math.Clamp(ay, context.Y, context.Bottom), Math.Clamp(bx, context.X, context.Right), Math.Clamp(by, context.Y, context.Bottom));
            switch (step.Choice("head-style", "open", "open", "closed", "filled"))
            {
                case "open":
                    paths.Add(ShapeGeometry.Path(step, [ax, ay, x1, y1, bx, by], closed: false));
                    break;
                case "closed":
                    paths.Add(ShapeGeometry.Path(step, [ax, ay, x1, y1, bx, by], closed: true));
                    break;
                default:
                    areas.Add(new ShapeArea([ax, ay, x1, y1, bx, by], step.Tone()));
                    paths.Add(ShapeGeometry.Path(step, [ax, ay, x1, y1, bx, by], closed: true));
                    break;
            }
        }

        return new ShapePlan(paths, areas);
    }
}

/// <summary>"fill": a solid tint over its box.</summary>
internal sealed class FillShape : IShape
{
    public string Name => "fill";

    public string Summary => "a solid tint over its box (tone: how dark, a share of the ink; 0.12)";

    public IReadOnlySet<string> Keys => OptionKeys.None;

    public void Check(ShapeStep step) => ShapeGeometry.NoValue(step);

    public ShapePlan Plan(ShapeStep step, ShapeContext context) =>
        context.Width <= 0 || context.Height <= 0 ? ShapePlan.Empty
        : new ShapePlan([], [new ShapeArea([context.X, context.Y, context.Right, context.Y, context.Right, context.Bottom, context.X, context.Bottom], step.Tone(0.12f))]);
}

/// <summary>"hatch": parallel lines across its box.</summary>
internal sealed class HatchShape : IShape
{
    private static readonly OptionKeys Own = new("spacing", "angle");

    public string Name => "hatch";

    public string Summary => "parallel lines across its box: spacing (stroke widths between them; 6), angle (degrees from the reading direction, up positive; 45)";

    public IReadOnlySet<string> Keys => Own;

    public void Check(ShapeStep step)
    {
        ShapeGeometry.NoValue(step);
        _ = step.Number("spacing", 6, 1, 10_000);
        _ = step.Number("angle", 45, -360, 360);
    }

    public ShapePlan Plan(ShapeStep step, ShapeContext context)
    {
        float spacing = step.Number("spacing", 6, 1, 10_000) * context.StrokeWidth * step.WidthFactor();
        double angle = step.Number("angle", 45, -360, 360) * Math.PI / 180;
        float dx = (context.RightToLeft ? -1 : 1) * (float)PortableMath.Cos(angle), dy = -(float)PortableMath.Sin(angle);
        float nx = -dy, ny = dx;
        float cx = context.X + context.Width / 2, cy = context.Y + context.Height / 2;
        // The lines across the box's extent along the normal, each cut to the box.
        float reach = (Math.Abs(nx) * context.Width + Math.Abs(ny) * context.Height) / 2, far = context.Width + context.Height;
        var paths = new List<ShapePath>();
        for (float t = -reach + spacing / 2; t < reach; t += spacing)
        {
            float px = cx + nx * t, py = cy + ny * t;
            float x0 = px - dx * far, y0 = py - dy * far, x1 = px + dx * far, y1 = py + dy * far;
            var (from, to) = ShapeGeometry.Inside(context, x0, y0, x1, y1);
            if ((to - from) * 2 * far > 0.5f)
            {
                paths.Add(ShapeGeometry.Path(step, [x0 + (x1 - x0) * from, y0 + (y1 - y0) * from, x0 + (x1 - x0) * to, y0 + (y1 - y0) * to], closed: false));
            }
        }

        return new ShapePlan(paths);
    }
}
