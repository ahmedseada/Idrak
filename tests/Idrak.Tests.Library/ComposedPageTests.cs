// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Data;
using Idrak.Abstraction.Testing;
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
}
