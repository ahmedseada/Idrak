// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Data;
using Idrak.Abstraction.Testing;
using Idrak.Imaging;

// Shapes drawn by a pen and the page's text set in the regions they make: the "shapes" page layout, the text binding,
// and what a shape's text says.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] ShapeGroup =
    [
        ("shapes: a grid page left to right and right to left: the rows and columns asked for, the first column on the start side, a line a row in the text, every region's text ink inside its region", ShapeGridPage),
        ("shapes: blanked regions draw nothing and are absent from the text, which is the lines less the blanked units; take=1:rest keeps a run of words in one region; blank-at picks the regions that may be blank; a text that would be set smaller than least-size is left out", ShapeBlanks),
        ("shapes: letter and plain draw the same bytes with no options and with the shapes layout's options; the pens' keys move only a letter's pen-drawn signature", ShapeOptionsLeaveLetters),
        ("shapes: a text says one of its words the page's fonts draw (those in the page's direction first), the page's own text or its initials when it has none, nothing when none can be drawn", OrnamentProfiles),
    ];

    internal static readonly PageContent LatinRows = new()
    {
        Body = ["1 alpha beta 10 20", "2 gamma delta 30 40", "3 epsilon zeta 50 60"],
        Language = "en",
    };

    internal static readonly PageContent ArabicRows = new()
    {
        Body = ["1 باب مدينة 10 20", "2 نهر سم 30 40", "3 كتب مدينة 50 60"],
        Language = "ar",
    };

    internal static IReadOnlyDictionary<string, string> PageOptionsOf(params (string Key, string Value)[] options) =>
        options.ToDictionary(o => o.Key, o => o.Value, StringComparer.OrdinalIgnoreCase);

    internal static ShapeStep ShapeStepOf(string name, string? value, params (string Key, string Value)[] options) =>
        new(name, value, options.Select(o => KeyValuePair.Create(o.Key, o.Value)));

    internal static void ShapeGridPage(Device device)
    {
        _ = device;
        var layout = new ShapesLayout();
        foreach (var (content, rtl) in new[] { (LatinRows, false), (ArabicRows, true) })
        {
            string where = rtl ? "right to left" : "left to right";
            var style = TestPageStyle() with { Options = PageOptionsOf(("shapes", "grid=3x5"), ("cell-align", "center")) };
            var page = layout.Render(content, style, 3);
            var regions = page.Blocks.Where(b => b.Kind == "region").ToList();
            Check(regions.Count == 15 && page.Blocks.Count(b => b.Kind == "shape") == 1, $"{where}: 15 regions of a 3 x 5 grid, one shape: {page.Description}");
            Check(regions.Select(r => MathF.Round(r.Y)).Distinct().Count() == 3 && regions.Select(r => MathF.Round(r.X)).Distinct().Count() == 5, $"{where}: three rows, five columns");
            Check(page.Text == string.Join("\n", content.Body), $"{where}: the text a line a row: {page.Text}");
            var first = regions.Single(r => r.Text == "1");
            var row = regions.Where(r => MathF.Abs(r.Y - first.Y) < 0.5f).ToList();
            Check(rtl ? first.X >= row.Max(r => r.X) : first.X <= row.Min(r => r.X), $"{where}: the first unit in the column on the {(rtl ? "right" : "left")}");
            Check(regions.Select(r => r.Text).SequenceEqual(content.Body.SelectMany(l => l.Split(' '))), $"{where}: the regions in reading order hold the units in order");

            // The text's ink: what a page with every region left empty does not have, all of it inside the regions.
            var empty = layout.Render(content, style with { Options = PageOptionsOf(("shapes", "grid=3x5"), ("cell-align", "center"), ("blank", "1")) }, 3);
            Check(empty.Text.Length == 0 && !empty.Blocks.Any(b => b.Kind == "region") && empty.Image.Height == page.Image.Height, $"{where}: every region left empty: no text");
            var (outside, inked) = TextInk(page, empty, regions);
            Check(outside == 0, $"{where}: {outside} pixels of text ink outside the regions");
            Check(inked.All(i => i), $"{where}: every region with text is inked");
        }
    }

    // The pixels where `page` differs from `blank` (the same page with no text) outside every region, and whether each region has such a pixel.
    internal static (int Outside, bool[] Inked) TextInk(RenderedPage page, RenderedPage blank, IReadOnlyList<PageBlock> regions)
    {
        var a = Planes(page.Image);
        var b = Planes(blank.Image);
        int width = page.Image.Width, outside = 0;
        var inked = new bool[regions.Count];
        for (int i = 0; i < a[0].Length; i++)
        {
            bool differs = false;
            for (int c = 0; c < a.Length && !differs; c++)
            {
                differs = a[c][i] != b[c][i];
            }

            if (!differs)
            {
                continue;
            }

            float x = i % width + 0.5f, y = i / width + 0.5f;
            int at = -1;
            for (int r = 0; r < regions.Count && at < 0; r++)
            {
                var region = regions[r];
                if (x >= region.X - 1 && x <= region.X + region.Width + 1 && y >= region.Y - 1 && y <= region.Y + region.Height + 1)
                {
                    at = r;
                }
            }

            if (at < 0)
            {
                outside++;
            }
            else
            {
                inked[at] = true;
            }
        }

        return (outside, inked);
    }

    internal static void ShapeBlanks(Device device)
    {
        _ = device;
        var layout = new ShapesLayout();
        var style = TestPageStyle();
        IReadOnlyDictionary<string, string> With(string blank) => PageOptionsOf(("shapes", "grid=3x5"), ("pen", "ballpoint"), ("blank", blank));
        var full = layout.Render(LatinRows, style with { Options = With("0") }, 11);
        var some = layout.Render(LatinRows, style with { Options = With("0.4") }, 11);
        var none = layout.Render(LatinRows, style with { Options = With("1") }, 11);
        var all = full.Blocks.Where(b => b.Kind == "region").ToList();
        var kept = some.Blocks.Where(b => b.Kind == "region").ToList();
        var blanked = all.Where(r => !kept.Any(k => k.X == r.X && k.Y == r.Y)).ToList();
        Check(all.Count == 15 && blanked.Count > 0 && kept.Count > 0 && kept.All(k => all.Any(r => r.X == k.X && r.Y == k.Y && r.Text == k.Text)),
            $"some regions blanked ({blanked.Count} of {all.Count}), the others hold what they held");
        string expected = string.Join("\n", all.GroupBy(r => r.Y).Select(g => string.Join(" ", g.Where(r => !blanked.Contains(r)).Select(r => r.Text))).Where(l => l.Length > 0));
        Check(some.Text == expected, $"the text is the lines less the blanked units: '{some.Text}', expected '{expected}'");
        var (outside, inked) = TextInk(some, none, blanked);
        Check(inked.All(i => !i), $"no text ink in a blanked region ({inked.Count(i => i)} inked)");
        var (keptOutside, keptInked) = TextInk(some, none, kept);
        Check(keptInked.All(i => i) && keptOutside == 0 && outside > 0, $"every kept region inked, no text ink outside them ({keptOutside} pixels)");

        var sequence = new SequenceBinding();
        ShapeRegion[] row = [.. Enumerable.Range(0, 5).Select(i => new ShapeRegion(0, i, i * 10, 0, 10, 10))];
        string[] line = ["7 Ahmed Ali Hassan 5 6 9"];
        var rest = sequence.Bind(line, row, sequence.Read(PageOptionsOf(("take", "1:rest"))), 1);
        Check(rest.Texts.SequenceEqual(["7", "Ahmed Ali Hassan", "5", "6", "9"]) && rest.Label.SequenceEqual(line), $"take=1:rest: {string.Join(" | ", rest.Texts)}");
        var picked = sequence.Bind(line, row, sequence.Read(PageOptionsOf(("take", "1:rest"), ("blank", "1"), ("blank-at", "3..5"))), 1);
        Check(picked.Texts.SequenceEqual(["7", "Ahmed Ali Hassan", "", "", ""]) && picked.Label.SequenceEqual(["7 Ahmed Ali Hassan"]), $"blank-at=3..5: {string.Join(" | ", picked.Texts)}");
        var box = sequence.Bind(["one two", "three"], [new ShapeRegion(0, 0, 0, 0, 10, 10)], TextBindingSettings.None, 1);
        Check(box.Texts.SequenceEqual(["one two"]) && box.Label.SequenceEqual(["one two"]), "a box takes its whole line; a line beyond the groups is drawn nowhere");

        var page = layout.Render(new PageContent { Body = [line[0]] }, style with { Options = PageOptionsOf(("shapes", "grid=1x5"), ("take", "1:rest")) }, 2);
        Check(page.Blocks.Any(b => b.Kind == "region" && b.Text == "Ahmed Ali Hassan") && page.Text == line[0], $"on a page, the run in one region: {page.Text}");

        var crowded = new PageContent { Body = ["Supercalifragilistic Supercalifragilistic short a b c"] };
        var strict = layout.Render(crowded, style with { Options = PageOptionsOf(("shapes", "grid=1x6"), ("least-size", "0.9")) }, 2);
        var loose = layout.Render(crowded, style with { Options = PageOptionsOf(("shapes", "grid=1x6"), ("least-size", "0")) }, 2);
        Check(strict.Text == "short a b c" && strict.Blocks.Count(b => b.Kind == "region") == 4 && loose.Text == crowded.Body[0],
            $"a word that would be set smaller than least-size is left out: '{strict.Text}'; with none, every word: '{loose.Text}'");
    }

    internal static void ShapeOptionsLeaveLetters(Device device)
    {
        _ = device;
        var options = PageOptionsOf(("shapes", "grid"), ("pen", "ballpoint"), ("bind", "sequence"), ("blank", "0.5"), ("blue-pen", "1"), ("cell-align", "end"));
        foreach (var (name, layout) in new (string, IPageLayout)[] { ("letter", new LetterLayout()), ("plain", new PlainLayout()) })
        {
            foreach (var content in new[] { ArabicLetter, EnglishLetter })
            {
                foreach (bool color in new[] { true, false })
                {
                    for (int seed = 1; seed <= 3; seed++)
                    {
                        var plain = layout.Render(content, TestPageStyle(color), seed);
                        var given = layout.Render(content, TestPageStyle(color) with { Options = options }, seed);
                        Check(Planes(plain.Image).Zip(Planes(given.Image)).All(p => p.First.SequenceEqual(p.Second)) && plain.Text == given.Text
                              && plain.Blocks.SequenceEqual(given.Blocks) && plain.Description == given.Description,
                            $"{name}, seed {seed}: the same page with the shapes layout's options");
                    }
                }
            }
        }

        // The pens' keys are the page's: a letter's signature pen (a ballpoint) reads them, and nothing else moves.
        foreach (var content in new[] { ArabicLetter, EnglishLetter })
        {
            var style = TestPageStyle() with { Emblem = 0, Stamp = 0 };
            var plain = new LetterLayout().Render(content, style, 2);
            var wobbly = new LetterLayout().Render(content, style with { Options = PageOptionsOf(("pen-wobble", "3")) }, 2);
            var signature = plain.Blocks.Concat(wobbly.Blocks).Where(b => b.Kind == "scribble").ToList();
            int outside = DifferingPixels(plain, wobbly).Count(p => !signature.Any(b => Within(p, b, 1)));
            Check(signature.Count == 2 && outside == 0 && !SamePage(plain, wobbly) && plain.Text == wobbly.Text,
                $"pen-wobble moves the pen-drawn signature only ({outside} pixels changed away from it)");
        }
    }

    // Whether a pixel is inside a block, give or take `pad` pixels.
    internal static bool Within((int X, int Y) p, PageBlock b, float pad) => p.X + 0.5f >= b.X - pad && p.X + 0.5f <= b.X + b.Width + pad && p.Y + 0.5f >= b.Y - pad && p.Y + 0.5f <= b.Y + b.Height + pad;

    internal static void OrnamentProfiles(Device device)
    {
        _ = device;
        // What a text says on a page.
        var face = SubsetFace();
        var latin = new ShapeLettering(new TextStyle(face, 20), "Northwind Trading Company", RightToLeft: false);
        var arabic = latin with { PageText = "شركة النور للتقنية", RightToLeft = true };
        float[] line = [0, 0, 100, 0];
        Check(ShapeDrawing.Say(new ShapeText(line, 10), latin) == "Northwind Trading Company" && ShapeDrawing.Say(new ShapeText(line, 10, initials: true), latin) == "NTC",
            "no words: the page's own text, or its initials in capitals");
        Check(ShapeDrawing.Say(new ShapeText(line, 10, initials: true), arabic) == "ش ا ل", $"right-to-left initials stand apart: {ShapeDrawing.Say(new ShapeText(line, 10, initials: true), arabic)}");
        string[] words = ["COPY", "نسخة", "DRAFT", "مسودة"];
        Check(Enumerable.Range(0, 4).All(pick => ShapeDrawing.Say(new ShapeText(line, 10, words, pick), latin) is "COPY" or "DRAFT")
              && Enumerable.Range(0, 4).All(pick => ShapeDrawing.Say(new ShapeText(line, 10, words, pick), arabic) is "نسخة" or "مسودة")
              && ShapeDrawing.Say(new ShapeText(line, 10, words, 1), latin) == "DRAFT", "words read in the page's direction first, the pick counted round");
        Check(ShapeDrawing.Say(new ShapeText(line, 10, ["漢字", "COPY"], 0), arabic) == "COPY" && ShapeDrawing.Say(new ShapeText(line, 10, ["漢字"]), latin) is null
              && ShapeDrawing.Say(new ShapeText(line, 10), latin with { PageText = null }) is null, "a word the fonts cannot draw is not said; with nothing to say, nothing is drawn");
        Throws<ArgumentException>(() => new ShapeText([0, 0, 1], 10), "an odd path");
        Throws<ArgumentException>(() => new ShapeText(line, 0), "a size of 0");
        Throws<ArgumentException>(() => new ShapeText(line, 10, [" "]), "an empty word");
    }
}
