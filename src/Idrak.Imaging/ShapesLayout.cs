// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Globalization;
using static Idrak.Imaging.PageDrawing;

namespace Idrak.Imaging;

/// <summary>
/// "shapes": shapes drawn by a pen, the body's lines set in the regions they make. It reads the style's options
/// (<see cref="PageStyle.Options"/>, once per page): <c>shapes</c> (a shape profile's name or text; "grid"), <c>pen</c>
/// ("ideal") and the pen's keys, <c>bind</c> ("sequence") and the binding's keys, <c>blue-pen</c> (the chance the pen and
/// the writing are blue; 0), <c>cell-align</c> (start, center, end or mixed; mixed), <c>stroke</c> (the stroke's width,
/// a share of the text size; 0.05..0.09) and <c>least-size</c> (the smallest a region's text is set, a share of the body's
/// size; 0.4). The profile is sampled from the page's seed, every step planned in the page inside its margins (paper kept
/// at the page's corners), the text bound to the regions, each region's text set in the page's font at the body's size or
/// smaller so its ink stays inside the region (a text that would need to be smaller than least-size is left out: its
/// region stays empty), the shapes stroked by the pen; the page grows taller when the rows need it. Only the body is
/// drawn; the text is what was drawn, a line a group of regions in reading order (the binding's label, less what was left
/// out). A profile that makes no region sets the lines in rows down the page.
/// </summary>
internal sealed class ShapesLayout : IPageLayout
{
    internal static readonly string[] Alignments = ["start", "center", "end", "mixed"];

    // The least width and height a region needs to hold text, in pixels.
    private const float LeastRegion = 3;

    public string Name => "shapes";

    public string Summary => "shapes drawn by a pen (a shape profile: lines, boxes, grids, circles, ...), the body's lines set in the regions they make";

    public RenderedPage Render(PageContent content, PageStyle style, int seed)
    {
        ArgumentNullException.ThrowIfNull(content);
        ArgumentNullException.ThrowIfNull(style);
        if (style.Fonts.Count == 0)
        {
            throw new ArgumentException("A page needs at least one font.", nameof(style));
        }

        // The page's settings, read once.
        var options = style.Options;
        var profile = ShapeProfiles.Resolve(Option(options, "shapes") ?? "grid");
        var pen = Pens.Get(Option(options, "pen") ?? Pens.Ideal);
        var penSettings = pen.Read(PageOptions.Only(options, pen.Keys));
        var binding = TextBindings.Get(Option(options, "bind") ?? TextBindings.Sequence);
        var bindSettings = binding.Read(PageOptions.Only(options, binding.Keys));
        float bluePen = Option(options, "blue-pen") is { } blueText ? PageOptions.Chance("Page setting blue-pen", blueText) : 0;
        string align = Option(options, "cell-align") is { } alignText ? PageOptions.Choice("Page setting cell-align", alignText, Alignments) : "mixed";
        var strokeShare = Option(options, "stroke") is { } strokeText ? PageOptions.Range("Page setting stroke", strokeText, 0.005f, 1) : (0.05f, 0.09f);
        float leastSize = Option(options, "least-size") is { } leastText ? PageOptions.Number("Page setting least-size", leastText, 0, 1) : 0.4f;

        // The page's choices: the same draws in the same order on every page.
        var c = new Choices(style, seed);
        bool rtl = Direction(content);
        int width = style.Width;
        float margin = width * c.Between(style.Margin);
        var font = c.Font();
        float size = c.Between(style.BodySize), spacing = c.Between(style.LineSpacing);
        var body = Style(style, font, size, spacing, content.Language);
        float strokeWidth = Math.Max(1f, size * c.Between(strokeShare));
        bool blue = c.Chance(bluePen);
        float topShare = c.Between(0.5f, 0.9f);
        var paper = Paper(c, style.Color);
        float darkness = c.Between(style.Ink);
        var blueInk = BlueInk(c);

        var lines = new List<string>(content.Body.Count);
        int mostUnits = 1;
        foreach (string raw in content.Body)
        {
            string line = raw.Trim();
            if (line.Length > 0)
            {
                lines.Add(line);
                mostUnits = Math.Max(mostUnits, line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).Length);
            }
        }

        float lineHeight = font.Metrics.LineHeight * body.Scale(font) * spacing;
        float stray = pen.Stray(penSettings, strokeWidth);
        // Paper at the page's edges, whatever the margin: a page cut to its parts reads its paper at the corner.
        float edge = 3 * strokeWidth + stray + 2;
        float left = Math.Max(margin, edge), top = Math.Max(margin * topShare, edge), bottomMargin = Math.Max(margin * 0.6f, edge);
        var area = new ShapeContext
        {
            X = left, Y = top, Width = Math.Max(1, width - 2 * left), Height = Math.Max(lineHeight, style.Height - top - bottomMargin),
            RightToLeft = rtl, GrowsDown = true, TextSize = size, LineHeight = lineHeight, Lines = Math.Max(1, lines.Count), MostUnits = mostUnits,
            StrokeWidth = strokeWidth,
        };

        int shapeSeed = Mix(seed, 1), penSeed = Mix(seed, 2), bindSeed = Mix(seed, 3), placeSeed = Mix(seed, 4);
        var pipeline = profile.Sample(shapeSeed);
        var (drawn, regions) = Plan(pipeline, area, shapeSeed);

        // Regions kept clear of a wobbling pen's reach, too small ones left out; with none at all, rows down the page.
        var usable = new List<ShapeRegion>(regions.Count);
        foreach (var r in regions)
        {
            var kept = r with { X = r.X + stray, Y = r.Y + stray, Width = r.Width - 2 * stray, Height = r.Height - 2 * stray };
            if (kept.Width >= LeastRegion && kept.Height >= LeastRegion)
            {
                usable.Add(kept);
            }
        }

        if (regions.Count == 0)
        {
            float row = lineHeight * 1.25f;
            for (int i = 0; i < lines.Count; i++)
            {
                usable.Add(new ShapeRegion(i, 0, area.X, area.Y + i * row, area.Width, row));
            }
        }

        var bound = binding.Bind(lines, usable, bindSettings, bindSeed);
        if (bound.Texts.Count != usable.Count)
        {
            throw new InvalidOperationException($"The text binding {binding.Name} gave {bound.Texts.Count} texts for {usable.Count} regions.");
        }

        // The page's height: every shape and region on it, with its bottom margin.
        float bottom = 0, widest = 1;
        foreach (var (_, plan) in drawn)
        {
            foreach (var path in plan.Paths)
            {
                bottom = Math.Max(bottom, path.Bounds().MaxY);
                widest = Math.Max(widest, path.Width);
            }

            foreach (var a in plan.Areas)
            {
                bottom = Math.Max(bottom, a.Bounds().MaxY);
            }
        }

        bottom += strokeWidth * widest / 2 + stray;
        foreach (var r in usable)
        {
            bottom = Math.Max(bottom, r.Y + r.Height);
        }

        int height = (int)Math.Ceiling(Math.Max(style.Height, bottom + bottomMargin));
        using var canvas = new PageCanvas(width, height);
        var layer = blue ? canvas.Blue : canvas.Ink;
        var blocks = new List<PageBlock>(drawn.Count + usable.Count);
        Draw(canvas, layer, drawn, pen, penSettings, penSeed, strokeWidth, blocks);

        // Each region's text, fitted inside it, placed as a hand would (aimed at its alignment, a little off).
        var place = new Random(placeSeed);
        var texts = new List<(ShapeRegion Region, string Text)>();
        for (int i = 0; i < usable.Count; i++)
        {
            double aim = place.NextDouble(), offX = place.NextDouble(), offY = place.NextDouble();       // drawn for every region
            string text = bound.Texts[i];
            if (text.Length == 0)
            {
                continue;
            }

            var region = usable[i];
            var (block, inkLeft, inkTop, inkWidth, inkHeight) = Fit(text, body, region.Width, region.Height, rtl);
            if (block.Style.Size < size * leastSize)
            {
                continue;                                           // too small to read: the region stays empty
            }

            float freeX = Math.Max(0, region.Width - inkWidth), freeY = Math.Max(0, region.Height - inkHeight);
            string at = align == "mixed" ? Alignments[Math.Min(2, (int)(aim * 3))] : align;
            float x = at switch { "start" => rtl ? freeX : 0, "end" => rtl ? 0 : freeX, _ => freeX / 2 };
            x = Math.Clamp(x + (float)(offX - 0.5) * Math.Min(freeX, size) * 0.3f, 0, freeX);
            float y = Math.Clamp(freeY / 2 + (float)(offY - 0.5) * freeY * 0.4f, 0, freeY);
            TextRenderer.Draw(block, region.X + x - inkLeft, region.Y + y - inkTop, layer, width, height);
            texts.Add((region, text));
        }

        // The text as drawn: a line a group, its regions' texts in reading order.
        texts.Sort((a, b) => a.Region.Group != b.Region.Group ? a.Region.Group.CompareTo(b.Region.Group) : a.Region.Index.CompareTo(b.Region.Index));
        var label = new List<string>();
        for (int i = 0; i < texts.Count; i++)
        {
            var (region, text) = texts[i];
            blocks.Add(new PageBlock("region", text, region.X, region.Y, region.Width, region.Height));
            if (i > 0 && texts[i - 1].Region.Group == region.Group)
            {
                label[^1] += " " + text;
            }
            else
            {
                label.Add(text);
            }
        }

        var image = canvas.Compose(paper, darkness, blueInk, style.Color);
        string described = profile.Name == "custom" ? "written" : profile.Name;
        string description = string.Create(CultureInfo.InvariantCulture,
            $"shapes: {described} [{pipeline}], pen {pen.Name}, bind {binding.Name}, {Describe(body)}, spacing {spacing:F2}, stroke {strokeWidth:F1}px, margin {margin:F0}px, {drawn.Count} shapes, {usable.Count} regions, {texts.Count} with text{(blue ? ", blue pen" : "")}");
        return new RenderedPage(image, string.Join("\n", label), blocks, description);
    }

    private static string? Option(IReadOnlyDictionary<string, string> options, string key) => options.TryGetValue(key, out var value) && value.Length > 0 ? value : null;

    // Every step's plan in order: in the page, or once inside each region the step before ended with (in=last); a step
    // whose plans make regions there takes those regions' place, one that makes none leaves them.
    private static (List<(ShapeStep Step, ShapePlan Plan)> Drawn, List<ShapeRegion> Regions) Plan(ShapePipeline pipeline, ShapeContext area, int shapeSeed)
    {
        var drawn = new List<(ShapeStep, ShapePlan)>();
        var regions = new List<ShapeRegion?>();
        var previous = new List<int>();
        int groups = 0;
        for (int s = 0; s < pipeline.Steps.Count; s++)
        {
            var step = pipeline.Steps[s];
            var shape = Shapes.Get(step.Name);
            var next = new List<int>();
            if (step.InLast())
            {
                for (int b = 0; b < previous.Count; b++)
                {
                    var parent = regions[previous[b]]!.Value;
                    var box = area with { X = parent.X, Y = parent.Y, Width = parent.Width, Height = parent.Height, GrowsDown = false, Seed = Mix(shapeSeed, 1 + s * 65_536 + b) };
                    var plan = shape.Plan(step, box.Place(step));
                    drawn.Add((step, plan));
                    if (plan.Regions.Count > 0)
                    {
                        regions[previous[b]] = null;
                        groups = Add(plan.Regions, groups, regions, next);
                    }
                    else
                    {
                        next.Add(previous[b]);
                    }
                }
            }
            else
            {
                var plan = shape.Plan(step, (area with { Seed = Mix(shapeSeed, 1 + s * 65_536) }).Place(step));
                drawn.Add((step, plan));
                groups = Add(plan.Regions, groups, regions, next);
            }

            previous = next;
        }

        var live = new List<ShapeRegion>(regions.Count);
        foreach (var r in regions)
        {
            if (r is { } region)
            {
                live.Add(region);
            }
        }

        return (drawn, live);
    }

    // A plan's regions, its groups numbered on from the page's in the order of their numbers.
    private static int Add(IReadOnlyList<ShapeRegion> made, int groups, List<ShapeRegion?> regions, List<int> next)
    {
        if (made.Count == 0)
        {
            return groups;
        }

        var numbers = new SortedSet<int>();
        foreach (var r in made)
        {
            numbers.Add(r.Group);
        }

        var renumber = new Dictionary<int, int>(numbers.Count);
        foreach (int g in numbers)
        {
            renumber[g] = groups + renumber.Count;
        }

        foreach (var r in made)
        {
            next.Add(regions.Count);
            regions.Add(r with { Group = renumber[r.Group] });
        }

        return groups + numbers.Count;
    }

    // The shapes, in order: each one's areas, then its paths through the pen; a block for each that drew something.
    private static void Draw(PageCanvas canvas, Span<float> layer, List<(ShapeStep Step, ShapePlan Plan)> drawn, IPen pen, PenSettings settings, int penSeed,
        float strokeWidth, List<PageBlock> blocks)
    {
        using var stroke = new PenStroke(256);
        using var outline = new OutlineBuilder();
        var path = ArrayPool<PenPoint>.Shared.Rent(64);
        try
        {
            int strokes = 0;
            foreach (var (_, plan) in drawn)
            {
                float x0 = float.MaxValue, y0 = float.MaxValue, x1 = float.MinValue, y1 = float.MinValue;
                foreach (var area in plan.Areas)
                {
                    outline.Clear();
                    outline.Polygon(area.Points.Span);
                    canvas.Fill(layer, outline.ToOutline(), area.Tone);
                    var (ax0, ay0, ax1, ay1) = area.Bounds();
                    (x0, y0, x1, y1) = (Math.Min(x0, ax0), Math.Min(y0, ay0), Math.Max(x1, ax1), Math.Max(y1, ay1));
                }

                foreach (var shapePath in plan.Paths)
                {
                    int count = shapePath.Count;
                    if (path.Length < count)
                    {
                        ArrayPool<PenPoint>.Shared.Return(path);
                        path = ArrayPool<PenPoint>.Shared.Rent(count);
                    }

                    float w = strokeWidth * shapePath.Width;
                    var points = shapePath.Points.Span;
                    for (int i = 0; i < count; i++)
                    {
                        path[i] = new PenPoint(points[2 * i], points[2 * i + 1], w);
                    }

                    pen.Stroke(path.AsSpan(0, count), shapePath.Closed, settings, Mix(penSeed, strokes++), stroke);
                    outline.Clear();
                    StrokeOutline.Add(outline, stroke.Points, stroke.Closed, shapePath.Dash.Span, w);
                    canvas.Fill(layer, outline.ToOutline(), stroke.MeanInk() * shapePath.Tone);
                    var (px0, py0, px1, py1) = shapePath.Bounds();
                    float reach = w / 2 + pen.Stray(settings, w);
                    (x0, y0, x1, y1) = (Math.Min(x0, px0 - reach), Math.Min(y0, py0 - reach), Math.Max(x1, px1 + reach), Math.Max(y1, py1 + reach));
                }

                if (x1 > x0 && y1 > y0)
                {
                    blocks.Add(new PageBlock("shape", "", Math.Max(0, x0), Math.Max(0, y0), Math.Min(canvas.Width, x1) - Math.Max(0, x0), Math.Min(canvas.Height, y1) - Math.Max(0, y0)));
                }
            }
        }
        finally
        {
            ArrayPool<PenPoint>.Shared.Return(path);
        }
    }

    // The text set on one line at the body's size, or smaller so that its ink (and its advance box), with a pixel round
    // it, fits the region; returns the block and the ink box's left, top, width and height relative to the block.
    private static (TextBlock Block, float Left, float Top, float Width, float Height) Fit(string text, TextStyle body, float width, float height, bool rtl)
    {
        var style = body;
        TextBlock block = null!;
        (float Left, float Top, float Width, float Height) ink = default;
        for (int attempt = 0; attempt < 6; attempt++)
        {
            block = TextLayout.Layout(text, style, float.PositiveInfinity, TextAlignment.Start, rtl);
            ink = InkBox(block);
            float scale = Math.Min(width / ink.Width, height / ink.Height);
            if (scale >= 1f)
            {
                break;
            }

            style = style with { Size = style.Size * scale * 0.99f };
        }

        return (block, ink.Left, ink.Top, ink.Width, ink.Height);
    }

    // The box around a block's advance and its glyphs' outlines (control points included), a pixel wider on every side
    // for the edge pixels the rasterizer touches.
    private static (float Left, float Top, float Width, float Height) InkBox(TextBlock block)
    {
        float x0 = 0, y0 = 0, x1 = block.Width, y1 = block.Height;
        foreach (var line in block.Lines)
        {
            foreach (var g in line.Glyphs)
            {
                var outline = g.Font.Outline(g.Glyph);
                if (outline.IsEmpty)
                {
                    continue;
                }

                var (minX, minY, maxX, maxY) = outline.Bounds();
                x0 = Math.Min(x0, g.X + minX * g.Scale);
                x1 = Math.Max(x1, g.X + maxX * g.Scale);
                y0 = Math.Min(y0, g.Y - maxY * g.Scale);
                y1 = Math.Max(y1, g.Y - minY * g.Scale);
            }
        }

        return (x0 - 1, y0 - 1, x1 - x0 + 2, y1 - y0 + 2);
    }
}
