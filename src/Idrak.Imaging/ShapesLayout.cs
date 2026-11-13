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
/// a share of the text size; 0.05..0.09), <c>pose</c> and its keys (the glyph pose every layout writes with; none) and <c>least-size</c> (the smallest a region's text is set, a share of the body's
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
        var glyphs = GlyphPainter.Read(options, Mix(seed, GlyphSeed));
        bool sections = Option(options, "sections") is { } sectionText && PageOptions.Choice("Page setting sections", sectionText, "none", "blank") == "blank";

        // The page's choices: the same draws in the same order on every page.
        var c = new Choices(style, seed);
        bool rtl = Direction(content);
        int width = style.Width;
        float margin = width * c.Between(style.Margin);
        var font = c.Font();
        float size = c.Between(style.BodySize), spacing = c.Between(style.LineSpacing);
        var body = Style(style, font, size, spacing, content.Language, Pairs(seed));
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
        if (sections || pipeline.Steps.Any(Composes))
        {
            return Composed(new Setup(content, style, profile, pipeline, pen, penSettings, binding, bindSettings, align, leastSize, glyphs, rtl, width, margin, body, size,
                spacing, strokeWidth, blue, paper, darkness, blueInk, lineHeight, stray, area, bottomMargin, shapeSeed, penSeed, bindSeed, placeSeed, sections));
        }

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
            var (block, inkLeft, inkTop, inkWidth, inkHeight) = Fit(text, body, region.Width, region.Height, rtl, glyphs);
            if (block.Style.Size < size * leastSize)
            {
                continue;                                           // too small to read: the region stays empty
            }

            float freeX = Math.Max(0, region.Width - inkWidth), freeY = Math.Max(0, region.Height - inkHeight);
            string at = align == "mixed" ? Alignments[Math.Min(2, (int)(aim * 3))] : align;
            float x = at switch { "start" => rtl ? freeX : 0, "end" => rtl ? 0 : freeX, _ => freeX / 2 };
            x = Math.Clamp(x + (float)(offX - 0.5) * Math.Min(freeX, size) * 0.3f, 0, freeX);
            float y = Math.Clamp(freeY / 2 + (float)(offY - 0.5) * freeY * 0.4f, 0, freeY);
            glyphs.Draw(block, region.X + x - inkLeft, region.Y + y - inkTop, layer, width, height);
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
            $"shapes: {described} [{pipeline}], pen {pen.Name}, bind {binding.Name}, {Describe(body)}, spacing {spacing:F2}, stroke {strokeWidth:F1}px, margin {margin:F0}px, {drawn.Count} shapes, {usable.Count} regions, {texts.Count} with text{(blue ? ", blue pen" : "")}{Paired(body)}{Posed(glyphs)}");
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
    // it and the most the page's glyph pose moves it, fits the region; returns the block and the ink box's left, top,
    // width and height relative to the block.
    private static (TextBlock Block, float Left, float Top, float Width, float Height) Fit(string text, TextStyle body, float width, float height, bool rtl, GlyphPainter glyphs)
    {
        var style = body;
        TextBlock block = null!;
        (float Left, float Top, float Width, float Height) ink = default;
        for (int attempt = 0; attempt < 6; attempt++)
        {
            block = TextLayout.Layout(text, style, float.PositiveInfinity, TextAlignment.Start, rtl);
            ink = InkBox(block, out float reach);
            if (!glyphs.IsIdentity)
            {
                float pad = glyphs.Reach(style.Size, reach);
                ink = (ink.Left - pad, ink.Top - pad, ink.Width + 2 * pad, ink.Height + 2 * pad);
            }

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
    // for the edge pixels the rasterizer touches; `reach`, the farthest an outline's box reaches from its pen origin.
    internal static (float Left, float Top, float Width, float Height) InkBox(TextBlock block, out float reach)
    {
        // The lines' advances (a wrapped block is as wide as its width, its lines no wider than they are).
        float x0 = 0, y0 = 0, x1 = 0, y1 = block.Height;
        foreach (var line in block.Lines)
        {
            x0 = Math.Min(x0, line.Left);
            x1 = Math.Max(x1, line.Left + line.Width);
        }

        reach = 0;
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
                float rx = Math.Max(Math.Abs(minX), Math.Abs(maxX)) * g.Scale, ry = Math.Max(Math.Abs(minY), Math.Abs(maxY)) * g.Scale;
                reach = Math.Max(reach, MathF.Sqrt(rx * rx + ry * ry));
            }
        }

        return (x0 - 1, y0 - 1, x1 - x0 + 2, y1 - y0 + 2);
    }

    // Whether a step needs the composed page: stacked, taking a share of the lines, holding paragraphs, or a text step.
    private static bool Composes(ShapeStep step) => step.Name == "text" || step.FlowsDown() || step.Has("lines") || step.HoldsParagraphs();

    // What a composed page is drawn from: the page's settings and choices, made as for any page.
    private sealed record Setup(PageContent Content, PageStyle Style, IShapeProfile Profile, ShapePipeline Pipeline, IPen Pen, PenSettings PenSettings,
        ITextBinding Binding, TextBindingSettings BindSettings, string Align, float LeastSize, GlyphPainter Glyphs, bool Rtl, int Width, float Margin,
        TextStyle Body, float Size, float Spacing, float StrokeWidth, bool Blue, (float R, float G, float B) Paper, float Darkness, (float R, float G, float B) BlueInk,
        float LineHeight, float Stray, ShapeContext Area, float BottomMargin, int ShapeSeed, int PenSeed, int BindSeed, int PlaceSeed, bool Sections);

    // A page of parts: steps placed in the page or stacked down it (flow=down, each its height from what it holds), the
    // page's lines shared between the steps that take text (lines=N|rest, or a section each with sections=blank), regions
    // holding one line of units or wrapped paragraphs. Its text is what was drawn, part after part: a line region's group
    // one line, a paragraph region's paragraphs one line each.
    private static RenderedPage Composed(Setup p)
    {
        // The page's lines in order, and its sections (runs between empty lines).
        var sections = new List<List<string>> { new() };
        foreach (string raw in p.Content.Body)
        {
            string line = raw.Trim();
            if (line.Length > 0)
            {
                sections[^1].Add(line);
            }
            else if (sections[^1].Count > 0)
            {
                sections.Add([]);
            }
        }

        if (sections.Count > 1 && sections[^1].Count == 0)
        {
            sections.RemoveAt(sections.Count - 1);
        }

        var all = sections.SelectMany(x => x).ToList();
        var steps = p.Pipeline.Steps;
        var slices = Share(p, steps, sections, all);

        // Every step planned in order: stacked ones down from the top of the page, the others where at and size put them.
        var drawn = new List<(ShapeStep Step, ShapePlan Plan)>();
        var regions = new List<ShapeRegion?>();
        var owners = new List<int>();
        var previous = new List<int>();
        int groups = 0;
        float cursor = p.Area.Y;
        for (int s = 0; s < steps.Count; s++)
        {
            var step = steps[s];
            var shape = Shapes.Get(step.Name);
            var next = new List<int>();
            if (step.InLast())
            {
                for (int b = 0; b < previous.Count; b++)
                {
                    var parent = regions[previous[b]]!.Value;
                    int owner = owners[previous[b]];
                    var slice = slices[owner] ?? [];
                    var box = p.Area with
                    {
                        X = parent.X, Y = parent.Y, Width = parent.Width, Height = parent.Height, GrowsDown = false, Seed = Mix(p.ShapeSeed, 1 + s * 65_536 + b),
                        Lines = Math.Max(1, slice.Count), MostUnits = Units(slice),
                    };
                    var plan = shape.Plan(step, box.Place(step));
                    drawn.Add((step, plan));
                    if (plan.Regions.Count > 0)
                    {
                        regions[previous[b]] = null;
                        groups = Add(plan.Regions, groups, regions, next);
                        while (owners.Count < regions.Count)
                        {
                            owners.Add(owner);
                        }
                    }
                    else
                    {
                        next.Add(previous[b]);
                    }
                }
            }
            else
            {
                var slice = slices[s] ?? [];
                var context = p.Area with { Lines = Math.Max(1, slice.Count), MostUnits = Units(slice), Seed = Mix(p.ShapeSeed, 1 + s * 65_536) };
                ShapePlan plan;
                if (step.FlowsDown())
                {
                    (plan, cursor) = Stack(p, step, shape, context, cursor, slice);
                }
                else
                {
                    plan = shape.Plan(step, context.Place(step));
                }

                drawn.Add((step, plan));
                groups = Add(plan.Regions, groups, regions, next);
                while (owners.Count < regions.Count)
                {
                    owners.Add(s);
                }
            }

            previous = next;
        }

        // Regions kept clear of the pen's reach, too small ones left out; with none at all, rows down the page.
        var usable = new List<(ShapeRegion Region, int Owner)>();
        for (int i = 0; i < regions.Count; i++)
        {
            if (regions[i] is { } r)
            {
                var kept = r with { X = r.X + p.Stray, Y = r.Y + p.Stray, Width = r.Width - 2 * p.Stray, Height = r.Height - 2 * p.Stray };
                if (kept.Width >= LeastRegion && kept.Height >= LeastRegion)
                {
                    usable.Add((kept, owners[i]));
                }
            }
        }

        var texts = new string[usable.Count];
        var paragraphs = new IReadOnlyList<string>?[usable.Count];
        Array.Fill(texts, "");
        if (regions.All(r => r is null))
        {
            float row = p.LineHeight * 1.25f;
            for (int i = 0; i < all.Count; i++)
            {
                usable.Add((new ShapeRegion(i, 0, p.Area.X, cursor + i * row, p.Area.Width, row), -1));
            }

            texts = [.. all];
            paragraphs = new IReadOnlyList<string>?[usable.Count];
        }
        else
        {
            // Each step's lines into its regions: its paragraph regions share them in order, else the binding puts them in its line regions.
            for (int s = 0; s < steps.Count; s++)
            {
                if (slices[s] is not { } slice)
                {
                    continue;
                }

                var mine = Enumerable.Range(0, usable.Count).Where(i => usable[i].Owner == s).ToList();
                var held = mine.Where(i => usable[i].Region.Paragraphs).ToList();
                if (held.Count > 0)
                {
                    for (int k = 0; k < held.Count; k++)
                    {
                        int from = slice.Count * k / held.Count, to = slice.Count * (k + 1) / held.Count;
                        paragraphs[held[k]] = slice.GetRange(from, to - from);
                    }

                    continue;
                }

                if (mine.Count > 0)
                {
                    var bound = p.Binding.Bind(slice, [.. mine.Select(i => usable[i].Region)], p.BindSettings, Mix(p.BindSeed, 1 + s));
                    if (bound.Texts.Count != mine.Count)
                    {
                        throw new InvalidOperationException($"The text binding {p.Binding.Name} gave {bound.Texts.Count} texts for {mine.Count} regions.");
                    }

                    for (int k = 0; k < mine.Count; k++)
                    {
                        texts[mine[k]] = bound.Texts[k];
                    }
                }
            }
        }

        // The page's height: every shape and region on it, with its bottom margin.
        float bottom = cursor, widest = 1;
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

        bottom += p.StrokeWidth * widest / 2 + p.Stray;
        foreach (var (r, _) in usable)
        {
            bottom = Math.Max(bottom, r.Y + r.Height);
        }

        int width = p.Width, height = (int)Math.Ceiling(Math.Max(p.Style.Height, bottom + p.BottomMargin));
        using var canvas = new PageCanvas(width, height);
        var layer = p.Blue ? canvas.Blue : canvas.Ink;
        var blocks = new List<PageBlock>(drawn.Count + usable.Count);
        Draw(canvas, layer, drawn, p.Pen, p.PenSettings, p.PenSeed, p.StrokeWidth, blocks);

        // Each region's text: one line fitted as a hand would place it, or paragraphs wrapped inside its edge.
        var place = new Random(p.PlaceSeed);
        var label = new List<(int Group, int Index, string Text, bool Paragraphs)>();
        for (int i = 0; i < usable.Count; i++)
        {
            double aim = place.NextDouble(), offX = place.NextDouble(), offY = place.NextDouble();       // drawn for every region
            var region = usable[i].Region;
            if (paragraphs[i] is { Count: > 0 } held)
            {
                float vertical = usable[i].Owner >= 0 ? steps[usable[i].Owner].Vertical() : 0.5f;
                var placed = FillParagraphs(p, region, held, justify: aim < p.Style.Justify, vertical, out int kept);
                if (placed is null)
                {
                    continue;                                       // nothing fits at the least size: the region stays empty
                }

                foreach (var (block, x, y) in placed)
                {
                    p.Glyphs.Draw(block, x, y, layer, width, height);
                }

                string text = string.Join("\n", held.Take(kept));
                label.Add((region.Group, region.Index, text, true));
                continue;
            }

            string unit = texts[i];
            if (unit.Length == 0)
            {
                continue;
            }

            var (one, inkLeft, inkTop, inkWidth, inkHeight) = Fit(unit, p.Body, region.Width, region.Height, p.Rtl, p.Glyphs);
            if (one.Style.Size < p.Size * p.LeastSize)
            {
                continue;
            }

            float freeX = Math.Max(0, region.Width - inkWidth), freeY = Math.Max(0, region.Height - inkHeight);
            string at = p.Align == "mixed" ? Alignments[Math.Min(2, (int)(aim * 3))] : p.Align;
            float ox = at switch { "start" => p.Rtl ? freeX : 0, "end" => p.Rtl ? 0 : freeX, _ => freeX / 2 };
            ox = Math.Clamp(ox + (float)(offX - 0.5) * Math.Min(freeX, p.Size) * 0.3f, 0, freeX);
            float oy = Math.Clamp(freeY / 2 + (float)(offY - 0.5) * freeY * 0.4f, 0, freeY);
            p.Glyphs.Draw(one, region.X + ox - inkLeft, region.Y + oy - inkTop, layer, width, height);
            label.Add((region.Group, region.Index, unit, false));
        }

        // The text as drawn, part after part: a line region's group one line, a paragraph region's paragraphs one line each.
        var regionOf = usable.Select(u => u.Region).ToList();
        label.Sort((a, b) => a.Group != b.Group ? a.Group.CompareTo(b.Group) : a.Index.CompareTo(b.Index));
        var lines = new List<string>();
        for (int i = 0; i < label.Count; i++)
        {
            var (group, index, text, held) = label[i];
            var region = regionOf.First(r => r.Group == group && r.Index == index);
            blocks.Add(new PageBlock("region", text, region.X, region.Y, region.Width, region.Height));
            if (held)
            {
                lines.AddRange(text.Split('\n'));
            }
            else if (i > 0 && !label[i - 1].Paragraphs && label[i - 1].Group == group)
            {
                lines[^1] += " " + text;
            }
            else
            {
                lines.Add(text);
            }
        }

        var image = canvas.Compose(p.Paper, p.Darkness, p.BlueInk, p.Style.Color);
        string described = p.Profile.Name == "custom" ? "written" : p.Profile.Name;
        string description = string.Create(CultureInfo.InvariantCulture,
            $"shapes: {described} [{p.Pipeline}], pen {p.Pen.Name}, bind {p.Binding.Name}, {Describe(p.Body)}, spacing {p.Spacing:F2}, stroke {p.StrokeWidth:F1}px, margin {p.Margin:F0}px, {drawn.Count} shapes, {usable.Count} regions, {label.Count} with text, composed{(p.Sections ? ", sections" : "")}{(p.Blue ? ", blue pen" : "")}{Paired(p.Body)}{Posed(p.Glyphs)}");
        return new RenderedPage(image, string.Join("\n", lines), blocks, description);
    }

    // The lines each step that takes text gets, in order: a section each (sections=blank), or lines=N, lines=rest (what
    // the others leave), and without lines= one a group when the step's groups are fixed, else the rest (the first such
    // step; any later one takes one line). Null for a step that takes none.
    private static List<string>?[] Share(Setup p, IReadOnlyList<ShapeStep> steps, List<List<string>> sections, List<string> all)
    {
        var wants = new (int Count, bool Remainder, bool Given)?[steps.Count];
        for (int s = 0; s < steps.Count; s++)
        {
            var step = steps[s];
            if (step.InLast())
            {
                continue;
            }

            var shape = Shapes.Get(step.Name);
            var seed = Mix(p.ShapeSeed, 1 + s * 65_536);
            var one = shape.Plan(step, (p.Area with { Lines = 1, MostUnits = 1, Seed = seed }).Place(step));
            if (one.Regions.Count == 0)
            {
                continue;
            }

            var seven = shape.Plan(step, (p.Area with { Lines = 7, MostUnits = 7, Seed = seed }).Place(step));
            int fixedGroups = Groups(one);
            bool follows = fixedGroups != Groups(seven) || one.Regions.Any(r => r.Paragraphs);
            wants[s] = step.Lines() is { } given ? (given.Count, given.Remainder, true) : follows ? (0, true, false) : (fixedGroups, false, false);
        }

        var slices = new List<string>?[steps.Count];
        if (p.Sections)
        {
            int k = 0;
            for (int s = 0; s < steps.Count; s++)
            {
                if (wants[s] is not null)
                {
                    slices[s] = k < sections.Count ? sections[k++] : [];
                }
            }

            return slices;
        }

        int rest = Array.FindIndex(wants, w => w is { Remainder: true, Given: true });
        if (rest < 0)
        {
            rest = Array.FindIndex(wants, w => w is { Remainder: true });
        }

        int Count(int s) => wants[s] is { } w ? s == rest ? 0 : w.Remainder ? 1 : w.Count : 0;
        int others = 0;
        for (int s = 0; s < steps.Count; s++)
        {
            others += Count(s);
        }

        int at = 0;
        for (int s = 0; s < steps.Count; s++)
        {
            if (wants[s] is null)
            {
                continue;
            }

            int n = Math.Min(s == rest ? Math.Max(0, all.Count - others) : Count(s), all.Count - at);
            slices[s] = all.GetRange(at, n);
            at += n;
        }

        return slices;
    }

    private static int Groups(ShapePlan plan) => plan.Regions.Select(r => r.Group).Distinct().Count();

    private static int Units(List<string> lines)
    {
        int most = 1;
        foreach (string line in lines)
        {
            most = Math.Max(most, line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).Length);
        }

        return most;
    }

    // A stacked step: at the cursor, the page's width (at and size within it), its height given (height=), or what its
    // paragraphs need, or its groups' lines; the cursor goes below what it drew and its gap.
    private static (ShapePlan Plan, float Cursor) Stack(Setup p, ShapeStep step, IShape shape, ShapeContext context, float cursor, List<string> slice)
    {
        var box = context with { Y = cursor, GrowsDown = true };
        ShapePlan At(float h) => shape.Plan(step, (box with { Height = h }).Place(step));
        float height;
        if (step.FlowHeight() is { } given)
        {
            height = Math.Min(given * p.LineHeight, Tallest(p));
        }
        else
        {
            var probe = At(p.LineHeight * 3);
            int held = probe.Regions.Count(r => r.Paragraphs);
            if (held > 0 && slice.Count > 0)
            {
                // The least height whose first paragraph region holds its share of the lines: the region's height less
                // what its text needs only grows with the step's height (a wider region needs fewer lines), so bisect it.
                // A round region is tried as it will be filled (kept clear of the pen, its lines shortened to the ellipse
                // where they sit, placed as valign puts them): whether the text fits at the body's size only grows with it.
                var share = slice.GetRange(0, Math.Max(1, slice.Count / held));
                var round = probe.Regions.First(r => r.Paragraphs).Round ? new RoundText(share, p.Body, p.Glyphs, p.Rtl) : null;
                float vertical = step.Vertical();
                float Spare(float h)
                {
                    var region = At(h).Regions.First(r => r.Paragraphs);
                    if (round is not null)
                    {
                        var kept = Kept(region, p.Stray);
                        return kept.Width >= LeastRegion && kept.Height >= LeastRegion && round.Place(kept, vertical) is not null ? 0 : -1;
                    }

                    return region.Height - (Need(p, share, region.Width - 2 * p.Stray) + 2 * p.Stray);
                }

                float low = p.LineHeight, high = p.LineHeight * 3;
                while (Spare(high) < 0 && high < Tallest(p))
                {
                    (low, high) = (high, Math.Min(high * 2, Tallest(p)));
                }

                if (Spare(high) >= 0)
                {
                    for (int k = 0; k < 20; k++)
                    {
                        float middle = (low + high) / 2;
                        if (Spare(middle) >= 0)
                        {
                            high = middle;
                        }
                        else
                        {
                            low = middle;
                        }
                    }

                    height = high;
                }
                else
                {
                    // Even the tallest does not hold it (a ring as wide as the page): the least height that gives the region its most.
                    float most = At(high).Regions.First(r => r.Paragraphs).Height;
                    low = p.LineHeight;
                    for (int k = 0; k < 20; k++)
                    {
                        float middle = (low + high) / 2;
                        if (At(middle).Regions.First(r => r.Paragraphs).Height >= most - 0.5f)
                        {
                            high = middle;
                        }
                        else
                        {
                            low = middle;
                        }
                    }

                    height = high;
                }
            }
            else
            {
                height = probe.Regions.Count > 0 ? p.LineHeight * 1.8f * Groups(probe) : p.LineHeight * 1.5f;
            }
        }

        var plan = At(height);
        float bottom = cursor + (plan.IsEmpty ? height : 0), widest = 0;
        foreach (var path in plan.Paths)
        {
            bottom = Math.Max(bottom, path.Bounds().MaxY);
            widest = Math.Max(widest, path.Width);
        }

        foreach (var area in plan.Areas)
        {
            bottom = Math.Max(bottom, area.Bounds().MaxY);
        }

        bottom += p.StrokeWidth * widest / 2 + (widest > 0 ? p.Stray : 0);
        foreach (var region in plan.Regions)
        {
            bottom = Math.Max(bottom, region.Y + region.Height);
        }

        return (plan, bottom + step.Gap() * p.LineHeight);
    }

    // The most a stacked step may be: a few pages.
    private static float Tallest(Setup p) => 8f * Math.Max(p.Style.Height, p.LineHeight);

    // The room kept round paragraphs' ink: a pixel and the most the glyph pose moves a glyph (its reach taken as 1.5 ems).
    private static float Room(Setup p, float size) => p.Glyphs.Reach(size, size * 1.5f) + 1;

    // The height paragraphs need at the body's size in a rectangle `width` wide.
    private static float Need(Setup p, IReadOnlyList<string> paragraphs, float width)
    {
        float room = Room(p, p.Size);
        float usable = width - 2 * room;
        if (!(usable > 1))
        {
            return float.MaxValue / 4;
        }

        var block = TextLayout.Layout(string.Join("\n", paragraphs), p.Body, usable, TextAlignment.Start, p.Rtl);
        return InkBox(block, out _).Height + 2 * room;
    }

    // A region kept clear of the pen's reach on every side, as the page fills it.
    private static ShapeRegion Kept(ShapeRegion r, float stray) =>
        r with { X = r.X + stray, Y = r.Y + stray, Width = r.Width - 2 * stray, Height = r.Height - 2 * stray };

    // Paragraphs set inside a region at the body's size, or smaller down to the least size; when even that does not fit,
    // the last paragraph is left out and the rest tried again. Null when none fits.
    private static List<(TextBlock Block, float X, float Y)>? FillParagraphs(Setup p, ShapeRegion region, IReadOnlyList<string> paragraphs, bool justify, float vertical, out int kept)
    {
        for (int count = paragraphs.Count; count > 0; count--)
        {
            var some = paragraphs.Take(count).ToList();
            for (float size = p.Size; size >= Math.Max(2f, p.Size * p.LeastSize) - 1e-3f; size *= 0.92f)
            {
                var style = p.Body with { Size = size };
                var placed = region.Round ? new RoundText(some, style, p.Glyphs, p.Rtl).Place(region, vertical) : Rectangle(p, region, some, style, justify, vertical);
                if (placed is not null)
                {
                    kept = count;
                    return placed;
                }
            }
        }

        kept = 0;
        return null;
    }

    // Paragraphs wrapped in a rectangular region, their ink (and the pose's reach) inside it, centred across and placed
    // down it at `vertical` (a share of the room left: 0 top, 0.5 centred, 1 bottom).
    private static List<(TextBlock, float, float)>? Rectangle(Setup p, ShapeRegion region, List<string> paragraphs, TextStyle style, bool justify, float vertical)
    {
        float room = Room(p, style.Size), width = region.Width - 2 * room;
        if (!(width > 1))
        {
            return null;
        }

        string text = string.Join("\n", paragraphs);
        var alignment = justify ? TextAlignment.Justify : TextAlignment.Start;
        var block = TextLayout.Layout(text, style, width, alignment, p.Rtl);
        var ink = InkBox(block, out float reach);
        float extra = p.Glyphs.Reach(style.Size, reach);
        if (ink.Width + 2 * extra > region.Width && width - (ink.Width + 2 * extra - region.Width) - 0.5f > 1)
        {
            // Ink past the advances (an overhanging letter): once more, narrower by what spilled.
            block = TextLayout.Layout(text, style, width - (ink.Width + 2 * extra - region.Width) - 0.5f, alignment, p.Rtl);
            ink = InkBox(block, out reach);
            extra = p.Glyphs.Reach(style.Size, reach);
        }

        if (ink.Width + 2 * extra > region.Width || ink.Height + 2 * extra > region.Height)
        {
            return null;
        }

        return [(block, region.X + (region.Width - ink.Width) / 2 - ink.Left, region.Y + (region.Height - ink.Height) * vertical - ink.Top)];
    }
}

/// <summary>
/// Paragraphs set in the ellipse inscribed in a region: each line as many words as fit the ellipse's width at its height
/// (its ink and the pose's reach included), centred across; the lines' block centred down the ellipse, or moved up or
/// down by a share of how far it can go and still fit. Each run of words is laid out and measured once, whatever region
/// it is tried in (the least ring that holds the text is searched with the same words many times).
/// </summary>
internal sealed class RoundText
{
    private readonly string[][] _words;
    private readonly TextStyle _style;
    private readonly GlyphPainter _glyphs;
    private readonly bool _rtl;
    private readonly float _lineHeight;
    private readonly float _spill;
    private readonly Dictionary<(int Paragraph, int Next, int Count), (TextBlock Block, float Left, float Top, float Width, float Height, float Need)> _runs = [];

    /// <summary>The paragraphs (each one line of the text) in <paramref name="style"/>, drawn by <paramref name="glyphs"/>.</summary>
    public RoundText(IReadOnlyList<string> paragraphs, TextStyle style, GlyphPainter glyphs, bool rtl)
    {
        _words = [.. paragraphs.Select(t => t.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries))];
        _style = style;
        _glyphs = glyphs;
        _rtl = rtl;
        _lineHeight = style.Font.Metrics.LineHeight * style.Scale(style.Font) * style.LineSpacing;

        // How far a line's ink reaches above and below its line box.
        var probe = ShapesLayout.InkBox(TextLayout.Layout(string.Join(" ", paragraphs), style, float.PositiveInfinity, TextAlignment.Start, rtl), out float probeReach);
        _spill = Math.Max(0, (probe.Height + 2 * glyphs.Reach(style.Size, probeReach) - _lineHeight) / 2);
    }

    /// <summary>The height of one line.</summary>
    public float LineHeight => _lineHeight;

    /// <summary>
    /// The lines placed in <paramref name="region"/>'s ellipse (each block with where its origin goes), the block at
    /// <paramref name="vertical"/>: 0.5 centred, 0 as high as it still fits, 1 as low, a share between; null when the
    /// text does not fit even centred.
    /// </summary>
    public List<(TextBlock Block, float X, float Y)>? Place(ShapeRegion region, float vertical)
    {
        var centred = Lines(region, 0);
        if (centred is null || vertical == 0.5f)
        {
            return centred;
        }

        // How far the block can move that way and still fit: the room shrinks the further it goes, so bisect it.
        float sign = vertical < 0.5f ? -1 : 1, low = 0, high = region.Height / 2;
        for (int k = 0; k < 16; k++)
        {
            float middle = (low + high) / 2;
            if (Lines(region, sign * middle) is not null)
            {
                low = middle;
            }
            else
            {
                high = middle;
            }
        }

        return Lines(region, sign * low * Math.Abs(vertical - 0.5f) * 2) ?? centred;
    }

    // The lines with the block's middle `shift` below the ellipse's centre; null when a line does not fit.
    private List<(TextBlock Block, float X, float Y)>? Lines(ShapeRegion region, float shift)
    {
        float cx = region.X + region.Width / 2, cy = region.Y + region.Height / 2, a = region.Width / 2, b = region.Height / 2;
        int lines = 1;
        for (int attempt = 0; attempt < 6; attempt++)
        {
            float top = cy - lines * _lineHeight / 2 + shift;
            var placed = new List<(TextBlock, float, float)>();
            int line = 0;
            for (int paragraph = 0; paragraph < _words.Length; paragraph++)
            {
                int next = 0, count = _words[paragraph].Length;
                while (next < count)
                {
                    float y0 = top + line * _lineHeight - _spill, y1 = top + (line + 1) * _lineHeight + _spill;
                    float d = Math.Max(Math.Abs(y0 - cy), Math.Abs(y1 - cy));
                    if (d >= b)
                    {
                        return null;
                    }

                    float room = 2 * a * MathF.Sqrt(1 - d * d / (b * b));
                    int take = 0;
                    for (int k = 1; next + k <= count && Run(paragraph, next, k).Need <= room; k++)
                    {
                        take = k;
                    }

                    if (take == 0)
                    {
                        return null;
                    }

                    var fit = Run(paragraph, next, take);
                    placed.Add((fit.Block, cx - fit.Width / 2 - fit.Left, top + line * _lineHeight + (_lineHeight - fit.Height) / 2 - fit.Top));
                    next += take;
                    line++;
                }
            }

            if (line == lines)
            {
                return placed;
            }

            lines = line;
        }

        return null;
    }

    // The words `next` .. `next + count` of a paragraph on one line: the block, its ink box and the width it needs.
    private (TextBlock Block, float Left, float Top, float Width, float Height, float Need) Run(int paragraph, int next, int count)
    {
        if (!_runs.TryGetValue((paragraph, next, count), out var run))
        {
            var block = TextLayout.Layout(string.Join(" ", _words[paragraph], next, count), _style, float.PositiveInfinity, TextAlignment.Start, _rtl);
            var ink = ShapesLayout.InkBox(block, out float reach);
            run = (block, ink.Left, ink.Top, ink.Width, ink.Height, ink.Width + 2 * _glyphs.Reach(_style.Size, reach));
            _runs[(paragraph, next, count)] = run;
        }

        return run;
    }
}
