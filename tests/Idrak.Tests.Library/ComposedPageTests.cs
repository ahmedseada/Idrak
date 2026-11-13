// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Data;
using Idrak.Abstraction.Testing;
using Idrak.Imaging;
using Idrak.Text;

// Pages of text and shapes together: parts stacked down the page (flow=down), the page's lines shared between the
// steps that take text (lines=N|rest, sections=blank), text steps and shapes holding wrapped paragraphs.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] ComposedPageGroup =
    [
        ("composed pages: the suites pass on the text step, on shapes holding paragraphs and on stacked, split and paragraph profiles", ComposedSuites),
        ("composed pages: a stacked page draws its parts top to bottom without overlap, all its ink inside its parts; its text is the intro, the rows and the closing line in order; left to right and right to left", ComposedStacked),
        ("composed pages: paragraphs inside a box and inside a circle stay inside the edge, the text exactly what was drawn; what does not fit at the least size is left out", ComposedInsideShapes),
        ("composed pages: a ring or an ellipse stacked down the page is the least that holds its paragraphs at the body's size (a pixel less does not), its text centred in it and inside its edge, the same page from the same seed; valign=top and bottom move the text up and down a ring of a given size, which keeps its size; left to right and right to left", ComposedRoundFit),
        ("composed pages: lines=N and lines=rest split the lines as asked; sections=blank splits at empty lines, read by the page settings, and is off by default (pages as before)", ComposedSplits),
    ];

    internal static readonly PageContent IntroRows = new()
    {
        Body = ["Weekly scores of the reading club are below.", "Each row is one member.", "1 alpha beta 10 20", "2 gamma delta 30 40", "3 epsilon zeta 50 60", "Thank you all for taking part."],
        Language = "en",
    };

    internal static readonly PageContent ArabicIntroRows = new()
    {
        Body = ["نتائج نادي القراءة لهذا الأسبوع في الجدول التالي.", "لكل عضو سطر واحد.", "1 باب مدينة 10 20", "2 نهر سم 30 40", "3 كتب مدينة 50 60", "شكرا لكم جميعا على المشاركة."],
        Language = "ar",
    };

    internal const string Stacked = "text,lines=2,flow=down,grid=auto x 5,flow=down,text,lines=1,flow=down";

    internal static void ComposedSuites(Device device)
    {
        _ = device;
        Check(Shapes.Names.Contains("text") && new[] { "text-grid-text", "framed", "circled" }.All(ShapeProfiles.Names.Contains), "the text step and the composed profiles are registered");
        Conformance.CheckShape("text", steps: [ShapeStepOf("text", null), ShapeStepOf("text", null, ("pad", "0.5"), ("lines", "2"))]).ThrowIfFailed();
        Conformance.CheckShape("box", steps: [ShapeStepOf("box", null, ("holds", "paragraphs"))]).ThrowIfFailed();
        Conformance.CheckShape("circle", steps: [ShapeStepOf("circle", null, ("holds", "paragraphs"), ("pad", "0.3"))]).ThrowIfFailed();
        Conformance.CheckShape("grid", steps: [ShapeStepOf("grid", "2x3", ("holds", "paragraphs"))]).ThrowIfFailed();
        var style = TestPageStyle();
        foreach (string shapes in new[] { "text-grid-text", "framed", "circled", Stacked, "text,lines=1,flow=down,ellipse,holds=paragraphs,size=0.8:1,flow=down,gap=1" })
        {
            Conformance.CheckPageLayout("shapes", style with { Options = PageOptionsOf(("shapes", shapes), ("pen", "ballpoint"), ("pose", "drift")) },
                [IntroRows, ArabicIntroRows, Newsletter]).ThrowIfFailed();
        }
    }

    internal static void ComposedStacked(Device device)
    {
        _ = device;
        var layout = PageLayouts.Get("shapes");
        foreach (var (content, rtl) in new[] { (IntroRows, false), (ArabicIntroRows, true) })
        {
            string where = rtl ? "right to left" : "left to right";
            var style = TestPageStyle() with { Options = PageOptionsOf(("shapes", Stacked)) };
            var page = layout.Render(content, style, 4);
            Check(page.Text == string.Join("\n", content.Body), $"{where}: the intro, the rows and the closing line, in order: {page.Text}");
            Check(SamePage(page, layout.Render(content, style, 4)), $"{where}: the same seed, the same page");
            var regions = page.Blocks.Where(b => b.Kind == "region").ToList();
            var grid = page.Blocks.Single(b => b.Kind == "shape");
            var intro = regions[0];
            var closing = regions[^1];
            Check(regions.Count == 2 + 15 && intro.Text == string.Join("\n", content.Body.Take(2)) && closing.Text == content.Body[^1], $"{where}: an intro, 15 cells, a closing paragraph");
            Check(intro.Y + intro.Height <= grid.Y && grid.Y + grid.Height <= closing.Y, $"{where}: the intro above the grid, the grid above the closing text");
            for (int i = 0; i < regions.Count; i++)
            {
                for (int j = i + 1; j < regions.Count; j++)
                {
                    Check(Overlap(regions[i], regions[j]) < 1, $"{where}: regions {i} and {j} overlap");
                }
            }

            // Every inked pixel inside one of the page's parts.
            var grey = Grey(page.Image);
            int width = page.Image.Width, outside = 0;
            for (int i = 0; i < grey.Length; i++)
            {
                float x = i % width + 0.5f, y = i / width + 0.5f;
                if (grey[i] < 0.75f && !page.Blocks.Any(b => x >= b.X - 1 && x <= b.X + b.Width + 1 && y >= b.Y - 1 && y <= b.Y + b.Height + 1))
                {
                    outside++;
                }
            }

            Check(outside == 0, $"{where}: {outside} inked pixels outside every part");
        }
    }

    internal static void ComposedInsideShapes(Device device)
    {
        _ = device;
        var layout = PageLayouts.Get("shapes");
        var empty = new PageContent { Body = [], Language = "en", RightToLeft = false };
        foreach (string shapes in new[] { "box,holds=paragraphs,size=0.8:0.5,at=0.1:0.1", "circle,holds=paragraphs,size=0.8:0.6,at=0.1:0.1" })
        {
            bool round = shapes.StartsWith("circle", StringComparison.Ordinal);
            var style = TestPageStyle() with { Options = PageOptionsOf(("shapes", shapes), ("pose", "drift")) };
            var page = layout.Render(Newsletter, style, 3);
            var bare = layout.Render(empty, style, 3);
            var region = page.Blocks.Single(b => b.Kind == "region");
            Check(page.Text == region.Text && Newsletter.Body.Take(region.Text.Split('\n').Length).SequenceEqual(region.Text.Split('\n')) && page.Text.Length > 0,
                $"{shapes}: the text is the paragraphs drawn, in order: {page.Text}");
            float cx = region.X + region.Width / 2, cy = region.Y + region.Height / 2, a = region.Width / 2 + 1, b = region.Height / 2 + 1;
            int outside = DifferingPixels(page, bare).Count(p =>
                round ? Math.Pow((p.X + 0.5 - cx) / a, 2) + Math.Pow((p.Y + 0.5 - cy) / b, 2) > 1
                      : p.X + 0.5 < region.X - 1 || p.X + 0.5 > region.X + region.Width + 1 || p.Y + 0.5 < region.Y - 1 || p.Y + 0.5 > region.Y + region.Height + 1);
            Check(outside == 0, $"{shapes}: {outside} pixels of text ink outside the {(round ? "circle" : "box")}'s region");
        }

        // A small ring cannot hold four paragraphs at a readable size: those that do not fit are left out, and out of the text.
        var small = layout.Render(Newsletter, TestPageStyle() with { Options = PageOptionsOf(("shapes", "circle,holds=paragraphs,size=0.25:0.12"), ("least-size", "0.8")) }, 3);
        var held = small.Blocks.Where(b => b.Kind == "region").Select(b => b.Text).ToList();
        Check(held.Count <= 1 && small.Text == string.Join("\n", held) && (held.Count == 0 || held[0].Split('\n').Length < Newsletter.Body.Count),
            $"a small ring keeps what fits: '{small.Text}'");
    }

    internal static void ComposedSplits(Device device)
    {
        _ = device;
        var layout = PageLayouts.Get("shapes");
        var six = new PageContent { Body = ["one", "two", "three", "four", "five", "six"], Language = "en" };
        var split = layout.Render(six, TestPageStyle() with { Options = PageOptionsOf(("shapes", "text,lines=1,flow=down,text,lines=rest,flow=down,text,lines=2,flow=down")) }, 1);
        var parts = split.Blocks.Where(b => b.Kind == "region").Select(b => b.Text).ToList();
        Check(parts.SequenceEqual(["one", "two\nthree\nfour", "five\nsix"]) && split.Text == string.Join("\n", six.Body), $"lines=1, rest, 2: {string.Join(" | ", parts)}");

        var withBreaks = new PageContent { Body = ["a b", "c d", "", "e f", "", "", "g h", "i j"], Language = "en" };
        var withoutBreaks = withBreaks with { Body = [.. withBreaks.Body.Where(l => l.Length > 0)] };
        var three = PageOptionsOf(("shapes", "text,flow=down,text,flow=down,text,flow=down"), ("sections", "blank"));
        var sectioned = layout.Render(withBreaks, TestPageStyle() with { Options = three }, 2);
        parts = sectioned.Blocks.Where(b => b.Kind == "region").Select(b => b.Text).ToList();
        Check(parts.SequenceEqual(["a b\nc d", "e f", "g h\ni j"]), $"sections=blank, a section a step: {string.Join(" | ", parts)}");
        var plainGrid = PageOptionsOf(("shapes", "grid"));
        Check(SamePage(layout.Render(withBreaks, TestPageStyle() with { Options = plainGrid }, 2), layout.Render(withoutBreaks, TestPageStyle() with { Options = plainGrid }, 2)),
            "without sections, empty lines change nothing");

        var style = new PageStyle { Fonts = [SubsetFace()], Width = 600, Height = 500, BodySize = (14, 16) };
        Check(SyntheticPages.Style(style, "sections=blank").Options["sections"] == "blank", "sections is a page setting");
        Throws<ArgumentException>(() => SyntheticPages.Style(style, "sections=lines"), "sections is none or blank");
        const string Answer = "First part, line one.\nFirst part, line two.\n\n\nSecond part.\n\nThird part.";
        var options = new SyntheticPageOptions { Style = style, Scans = ["none"], Layouts = ["shapes"], Shapes = ["text,flow=down,text,flow=down,text,flow=down"] };
        var made = SyntheticPages.Make(Answer, options with { Style = SyntheticPages.Style(style, "sections=blank") });
        Check(made.Text == "First part, line one.\nFirst part, line two.\nSecond part.\nThird part.", $"a page from sections: {made.Text}");
        var asBefore = SyntheticPages.Make(Answer, options with { Layouts = ["plain"] });
        var noBreaks = SyntheticPages.Make(Answer.Replace("\n\n\n", "\n", StringComparison.Ordinal).Replace("\n\n", "\n", StringComparison.Ordinal), options with { Layouts = ["plain"] });
        Check(asBefore.Image.Pixels.SequenceEqual(noBreaks.Image.Pixels) && asBefore.Text == noBreaks.Text, "sections off: empty lines dropped, the page as before");
    }

    internal static void ComposedRoundFit(Device device)
    {
        _ = device;
        var layout = PageLayouts.Get("shapes");
        var face = SubsetFace();
        var still = new GlyphPainter(GlyphPoses.Get(GlyphPoses.None), GlyphPoseSettings.None, 1);
        var arabic = new PageContent
        {
            Body = ["نشرة الحي لشهر مارس.", "يبدأ موسم الزراعة يوم 4 أبريل، فأحضروا القفازات وصديقا.", "صناديق السماد الجديدة بجوار البوابة الشمالية."],
            Language = "ar",
        };
        foreach (var (content, rtl) in new[] { (Newsletter, false), (arabic, true) })
        {
            var empty = content with { Body = [], RightToLeft = rtl };            // drawn as the text is, its ring where the text's is
            foreach (string shapes in new[] { "circle,holds=paragraphs,flow=down,pad=0.3", "ellipse,radii=1:0.7,holds=paragraphs,flow=down,pad=0.3" })
            {
                bool circle = shapes.StartsWith("circle", StringComparison.Ordinal);
                string where = $"{(rtl ? "right to left" : "left to right")}, {shapes}";
                var style = TestPageStyle(color: false) with { Fonts = [face], BodySize = (20, 20), LineSpacing = (1.2f, 1.2f), Options = PageOptionsOf(("shapes", shapes)) };
                var page = layout.Render(content, style, 5);
                Check(SamePage(page, layout.Render(content, style, 5)), $"{where}: the same seed, the same page");
                Check(page.Text == string.Join("\n", content.Body), $"{where}: every paragraph held: {page.Text}");

                // The least that holds it: the paragraphs fit the region at the body's size, and not a pixel less of it
                // (a ring's height and width shrink together, a page-wide ellipse's height alone).
                var block = page.Blocks.Single(b => b.Kind == "region");
                var region = new ShapeRegion(0, 0, block.X, block.Y, block.Width, block.Height) { Paragraphs = true, Round = true };
                var body = PageDrawing.Style(style, face, 20, 1.2f, content.Language);
                var text = new RoundText(content.Body, body, still, rtl);
                float shrinkX = circle ? 1 : 0;
                var less = region with { X = region.X + shrinkX / 2, Y = region.Y + 0.5f, Width = region.Width - shrinkX, Height = region.Height - 1 };
                Check(text.Place(region, 0.5f) is not null && text.Place(less, 0.5f) is null,
                    $"{where}: the region ({region.Width:F1} x {region.Height:F1}) holds the text at the body's size and a pixel less does not");

                var (inside, mx, my) = InkInStackedRound(page, block, page.Blocks.Single(b => b.Kind == "shape"));
                Check(inside == 0, $"{where}: {inside} pixels of text ink outside the edge");

                // Centred: each line's box is centred on the ellipse's centre and its ink centred across it, so the
                // ink's middle is off the centre only by where the letters sit in their line box (between the font's
                // ascent and descent): a quarter of a line at most down, and a tenth of the size across (side bearings).
                float cx = block.X + block.Width / 2, cy = block.Y + block.Height / 2;
                Check(Math.Abs(my - cy) <= 0.25f * text.LineHeight && Math.Abs(mx - cx) <= 2f,
                    $"{where}: the ink's middle ({mx:F1}, {my:F1}) is the region's ({cx:F1}, {cy:F1}) within ({2f:F1}, {0.25f * text.LineHeight:F1})");
            }

            // A ring of a given size keeps it; valign moves its text up and down inside it.
            var middles = new List<float>();
            foreach (string at in new[] { "top", "center", "bottom" })
            {
                string shapes = $"circle,holds=paragraphs,at=0.1:0.05,size=0.8:0.6,pad=0.3,valign={at}";
                var style = TestPageStyle(color: false) with { Fonts = [face], BodySize = (20, 20), LineSpacing = (1.2f, 1.2f), Options = PageOptionsOf(("shapes", shapes)) };
                var page = layout.Render(content, style, 5);
                var block = page.Blocks.Single(b => b.Kind == "region");
                var (inside, _, my, _, _) = InkInRound(page, layout.Render(empty, style, 5), block);
                Check(inside == 0 && page.Text == string.Join("\n", content.Body), $"{at}: every paragraph, {inside} pixels of ink outside the edge");
                middles.Add(my);
            }

            Check(middles[0] < middles[1] - 5 && middles[1] < middles[2] - 5, $"valign top, center, bottom: the text's middles {string.Join(", ", middles.Select(m => m.ToString("F1", System.Globalization.CultureInfo.InvariantCulture)))} go down");
        }

        // A box's paragraphs too: valign=top puts them at its top.
        var boxes = new List<float>();
        foreach (string at in new[] { "top", "0.5", "1" })
        {
            var style = TestPageStyle(color: false) with { Fonts = [face], BodySize = (20, 20), Options = PageOptionsOf(("shapes", $"box,holds=paragraphs,at=0.1:0.1,size=0.8:0.6,valign={at}")) };
            var page = layout.Render(Newsletter, style, 5);
            boxes.Add(InkInRound(page, layout.Render(Newsletter with { Body = [] }, style, 5), page.Blocks.Single(b => b.Kind == "region")).MiddleY);
        }

        Check(boxes[0] < boxes[1] - 5 && boxes[1] < boxes[2] - 5, $"a box's valign top, 0.5, 1 go down: {string.Join(", ", boxes)}");
        Check(Math.Abs(ShapeStepOf("circle", null).Vertical() - 0.5f) < 1e-6f && ShapeStepOf("circle", null, ("valign", "top")).Vertical() == 0
            && ShapeStepOf("circle", null, ("valign", "bottom")).Vertical() == 1 && Math.Abs(ShapeStepOf("box", null, ("valign", "0.3")).Vertical() - 0.3f) < 1e-6f,
            "valign: center by default, top 0, bottom 1, a share");
        Throws<ArgumentException>(() => ShapeStepOf("box", null, ("valign", "up")).CheckCommon(), "valign is center, top, bottom or a share");
    }

    // A stacked ring's text ink, from the page alone (without its text the ring would be sized for none): the dark
    // pixels off the ring's own stroke (its box the shape's block, a band of 4 pixels round its curve) outside the
    // ellipse inscribed in the region (a pixel of slack), and the middle of the box of those inside it.
    private static (int Outside, float MiddleX, float MiddleY) InkInStackedRound(RenderedPage page, PageBlock region, PageBlock ring)
    {
        var grey = Grey(page.Image);
        int width = page.Image.Width, outside = 0, left = int.MaxValue, top = int.MaxValue, right = int.MinValue, bottom = int.MinValue;
        float cx = region.X + region.Width / 2, cy = region.Y + region.Height / 2, a = region.Width / 2 + 1, b = region.Height / 2 + 1;
        float rcx = ring.X + ring.Width / 2, rcy = ring.Y + ring.Height / 2, rx = ring.Width / 2, ry = ring.Height / 2;
        for (int i = 0; i < grey.Length; i++)
        {
            if (grey[i] >= 0.75f)
            {
                continue;
            }

            int x = i % width, y = i / width;
            if (Math.Pow((x + 0.5 - cx) / a, 2) + Math.Pow((y + 0.5 - cy) / b, 2) <= 1)
            {
                (left, top, right, bottom) = (Math.Min(left, x), Math.Min(top, y), Math.Max(right, x), Math.Max(bottom, y));
                continue;
            }

            double off = Math.Abs(Math.Sqrt(Math.Pow((x + 0.5 - rcx) / rx, 2) + Math.Pow((y + 0.5 - rcy) / ry, 2)) - 1) * Math.Min(rx, ry);
            if (off > 4)
            {
                outside++;
            }
        }

        return right < left ? (outside, float.NaN, float.NaN) : (outside, (left + right + 1) / 2f, (top + bottom + 1) / 2f);
    }

    // The text's ink (the pixels that differ from the page drawn without text): how many lie outside the ellipse
    // inscribed in the region (a pixel of slack), the middle of their box and its height and width.
    private static (int Outside, float MiddleX, float MiddleY, float Width, float Height) InkInRound(RenderedPage page, RenderedPage bare, PageBlock region)
    {
        float cx = region.X + region.Width / 2, cy = region.Y + region.Height / 2, a = region.Width / 2 + 1, b = region.Height / 2 + 1;
        int outside = 0, left = int.MaxValue, top = int.MaxValue, right = int.MinValue, bottom = int.MinValue;
        foreach (var (x, y) in DifferingPixels(page, bare))
        {
            if (Math.Pow((x + 0.5 - cx) / a, 2) + Math.Pow((y + 0.5 - cy) / b, 2) > 1)
            {
                outside++;
            }

            (left, top, right, bottom) = (Math.Min(left, x), Math.Min(top, y), Math.Max(right, x), Math.Max(bottom, y));
        }

        return right < left ? (outside, float.NaN, float.NaN, 0, 0) : (outside, (left + right + 1) / 2f, (top + bottom + 1) / 2f, right + 1 - left, bottom + 1 - top);
    }
}
