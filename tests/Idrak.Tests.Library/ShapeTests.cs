// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Data;
using Idrak.Abstraction.Testing;
using Idrak.Imaging;

// Shapes drawn by a pen and the page's text set in the regions they make: the "shapes" page layout and its contracts
// (shapes, shape profiles, pens, text bindings), each checked by its suite on every built-in.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] ShapeGroup =
    [
        ("shapes: the contract suites pass on every built-in shape, pen (ideal draws its path exactly) and text binding, and on the letter, plain and shapes layouts", ShapeSuites),
        ("shapes: a grid page left to right and right to left: the rows and columns asked for, the first column on the start side, a line a row in the text, every region's text ink inside its region", ShapeGridPage),
        ("shapes: blanked regions draw nothing and are absent from the text, which is the lines less the blanked units; take=1:rest keeps a run of words in one region; blank-at picks the regions that may be blank; a text that would be set smaller than least-size is left out", ShapeBlanks),
        ("shapes: the same seed draws the same bytes; a grid draws each line once; profiles by name and as text (ranges, chances, in=last), pipelines as text and JSON; unknown shapes, options and bad values are refused; an app's pen replaces the library's of its name only", ShapeProfilesAndPipelines),
        ("shapes: letter and plain draw the same bytes with no options and with the shapes layout's options; the pens' keys move only a letter's pen-drawn signature", ShapeOptionsLeaveLetters),
        ("shapes: every emblem, stamp and signature profile the library registers draws inside its box (paths, areas and texts) on many seeds, each of its steps passing the shape suite; a text says one of its words the page's fonts draw (those in the page's direction first), the page's own text or its initials when it has none, nothing when none can be drawn", OrnamentProfiles),
        ("shapes: a plan's texts written as vector content (DocumentShapes with a lettering): every emblem and stamp profile's texts, upright and turned, read back with the library's reader as what each says, glyph by glyph in reading order; the written page drawn back by the library's drawer is close to the plan drawn on a page with the ideal pen (paths, areas and the texts turned onto their paths)", ShapeTextsWritten),
        ("shapes: a letter draws each emblem and stamp profile asked for, named in its description, its ink only inside its block, the text unchanged; the signature is drawn by the pen asked over the name, blue or black, only inside its block; the same seed the same page; their settings leave a page without them byte for byte as it was, and bad ones are refused naming them", LetterOrnamentPages),
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

    internal static void ShapeSuites(Device device)
    {
        _ = device;
        var steps = new Dictionary<string, ShapeStep[]>
        {
            ["line"] = [ShapeStepOf("line", null), ShapeStepOf("line", null, ("length", "0.8"), ("angle", "30")), ShapeStepOf("line", null, ("from", "0.1:0.9"), ("to", "0.9:0.1"))],
            ["polyline"] = [ShapeStepOf("polyline", null, ("points", "0:0:0.5:1:1:0"))],
            ["polygon"] = [ShapeStepOf("polygon", null), ShapeStepOf("polygon", "3"), ShapeStepOf("polygon", null, ("points", "0:0:1:0:0.5:1"))],
            ["box"] = [ShapeStepOf("box", null), ShapeStepOf("box", null, ("radius", "0.2")), ShapeStepOf("box", null, ("sides", "top:end:bottom"), ("radius", "0.1"))],
            ["grid"] =
            [
                ShapeStepOf("grid", null), ShapeStepOf("grid", "3x4", ("widths", "1:2:1:3")), ShapeStepOf("grid", "auto x 5", ("heavy-every", "2"), ("heavy-from", "1"), ("border", "bottom"), ("inner", "rows")),
                ShapeStepOf("grid", "2x3", ("heights", "1:2"), ("pad", "0.3"), ("widths", "3:1:1")),
            ],
            ["circle"] = [ShapeStepOf("circle", null), ShapeStepOf("circle", null, ("radius", "0.5"))],
            ["ellipse"] = [ShapeStepOf("ellipse", null), ShapeStepOf("ellipse", null, ("radii", "0.6:1"))],
            ["arc"] = [ShapeStepOf("arc", null), ShapeStepOf("arc", null, ("start", "-45"), ("end", "225"))],
            ["arrow"] = [ShapeStepOf("arrow", null), ShapeStepOf("arrow", null, ("head-style", "filled"), ("from", "1:1"), ("to", "0:0")), ShapeStepOf("arrow", null, ("head-style", "closed"))],
            ["fill"] = [ShapeStepOf("fill", null)],
            ["hatch"] = [ShapeStepOf("hatch", null), ShapeStepOf("hatch", null, ("angle", "-30"), ("spacing", "3"))],
            ["ring-text"] =
            [
                ShapeStepOf("ring-text", null), ShapeStepOf("ring-text", null, ("side", "bottom"), ("words", "ONE:TWO"), ("font-size", "0.3")),
                ShapeStepOf("ring-text", null, ("radius", "0.6"), ("span", "250"), ("initials", "on")),
            ],
            ["lettering"] = [ShapeStepOf("lettering", null), ShapeStepOf("lettering", null, ("words", "COPY"), ("font-size", "0.9")), ShapeStepOf("lettering", null, ("initials", "on"))],
            ["star"] = [ShapeStepOf("star", null), ShapeStepOf("star", "8", ("inner", "0.7"))],
            ["shield"] = [ShapeStepOf("shield", null), ShapeStepOf("shield", null, ("shoulder", "0.3"))],
            ["scribble"] =
            [
                ShapeStepOf("scribble", null), ShapeStepOf("scribble", null, ("letters", "12"), ("slant", "-20"), ("flourish", "under")), ShapeStepOf("scribble", null, ("flourish", "loop")),
            ],
        };
        Check(steps.Keys.All(Shapes.Names.Contains), $"the built-in shapes: {string.Join(", ", Shapes.Names)}");
        foreach (var (name, given) in steps)
        {
            Conformance.CheckShape(name, steps: given).ThrowIfFailed();
        }

        Conformance.CheckPen(Pens.Ideal, exact: true).ThrowIfFailed();
        Conformance.CheckPen("ballpoint", settings: [PageOptionsOf(("pen-wobble", "2..3"), ("pen-tilt", "4"), ("pen-overshoot", "3"), ("pen-pressure", "0.6"))]).ThrowIfFailed();
        Conformance.CheckTextBinding(TextBindings.Sequence, settings:
            [PageOptionsOf(("take", "1:rest:2")), PageOptionsOf(("blank", "0.5"), ("blank-at", "2..4")), PageOptionsOf(("take", "2"), ("blank", "1"))]).ThrowIfFailed();

        var style = TestPageStyle();
        Conformance.CheckPageLayout("letter", style, [ArabicLetter, EnglishLetter]).ThrowIfFailed();
        Conformance.CheckPageLayout("plain", style, [ArabicLetter, EnglishLetter]).ThrowIfFailed();
        foreach (var options in new[]
        {
            PageOptionsOf(("shapes", "grid"), ("pen", "ballpoint")),
            PageOptionsOf(("shapes", "boxed"), ("blank", "0.3"), ("blue-pen", "1")),
            PageOptionsOf(("shapes", "ruled"), ("cell-align", "start")),
            PageOptionsOf(("shapes", "none")),
        })
        {
            Conformance.CheckPageLayout("shapes", style with { Options = options }, [LatinRows, ArabicRows, EnglishLetter]).ThrowIfFailed();
        }
    }

    internal static void ShapeGridPage(Device device)
    {
        _ = device;
        var layout = PageLayouts.Get("shapes");
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
        var layout = PageLayouts.Get("shapes");
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

        var sequence = TextBindings.Get(TextBindings.Sequence);
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

    internal static void ShapeProfilesAndPipelines(Device device)
    {
        _ = device;
        var layout = PageLayouts.Get("shapes");
        var options = PageOptionsOf(("shapes", "box@0.7,grid=auto x 3..5,in=last,heavy-every=2,heavy-width=1.5..3"), ("pen", "ballpoint"), ("blank", "0.2"), ("blue-pen", "0.5"));
        var style = TestPageStyle() with { Options = options };
        var once = layout.Render(LatinRows, style, 5);
        var twice = layout.Render(LatinRows, style, 5);
        Check(Planes(once.Image).Zip(Planes(twice.Image)).All(p => p.First.SequenceEqual(p.Second)) && once.Text == twice.Text && once.Description == twice.Description,
            $"the same seed, the same bytes ({once.Description})");
        var others = Enumerable.Range(1, 6).Select(s => layout.Render(LatinRows, style, s).Description).Distinct().Count();
        Check(others >= 4, $"six seeds, {others} different pages");

        // A grid's lines: the border once, every inner line once across it.
        var grid = Shapes.Get("grid");
        var context = new ShapeContext { X = 10, Y = 20, Width = 400, Height = 300, Lines = 3, MostUnits = 4, StrokeWidth = 2 };
        var plan = grid.Plan(ShapeStepOf("grid", "3x4"), context);
        Check(plan.Paths.Count == 1 + 2 + 3 && plan.Paths[0].Closed && plan.Regions.Count == 12, $"3 x 4: a border and 2 + 3 inner lines: {plan}");
        var open = grid.Plan(ShapeStepOf("grid", "3x4", ("border", "off")), context);
        var horizontal = open.Paths.Where(p => p.Points.Span[1] == p.Points.Span[3]).Select(p => p.Points.Span[1]).ToList();
        var vertical = open.Paths.Where(p => p.Points.Span[0] == p.Points.Span[2]).Select(p => p.Points.Span[0]).ToList();
        Check(open.Paths.Count == 5 && horizontal.Distinct().Count() == 2 && vertical.Distinct().Count() == 3, $"no border: each inner line once: {open}");
        var heavy = grid.Plan(ShapeStepOf("grid", "2x6", ("heavy-every", "2"), ("heavy-from", "1"), ("heavy-width", "3")), context);
        Check(heavy.Paths.Count(p => p.Width == 3) == 3, $"heavier lines after columns 1, 3 and 5: {string.Join(", ", heavy.Paths.Select(p => p.Width))}");

        // Profiles: the built-in looks, and profiles written as text (parsed once), sampled from a seed.
        Check(new[] { "none", "grid", "ruled", "boxed" }.All(ShapeProfiles.Names.Contains), $"profiles: {string.Join(", ", ShapeProfiles.Names)}");
        Check(ShapeProfiles.Resolve("none").Sample(1).IsEmpty && ShapeProfiles.Resolve("grid").Sample(1).ToString() == "grid=auto x auto", "none and grid");
        const string Written = "grid=auto x 3..5@1,widths=1:2..3:1,heavy-width=1.5..3,line@0";
        var written = ShapeProfiles.Resolve(Written);
        Check(ReferenceEquals(written, ShapeProfiles.Resolve(Written)), "a profile written as text is parsed once");
        var samples = Enumerable.Range(0, 30).Select(written.Sample).ToList();
        Check(samples.All(p => p.Steps.Count == 1 && p.Steps[0].Name == "grid"), "a step at chance 0 is never taken");
        var columns = samples.Select(p => p.Steps[0].Value!).Distinct().ToList();
        Check(columns.All(v => v is "auto x 3" or "auto x 4" or "auto x 5") && columns.Count == 3, $"a whole value's range drawn whole: {string.Join(", ", columns)}");
        Check(samples.Select(p => p.Steps[0].Option("heavy-width")).Distinct().Count() > 3 && samples.All(p => p.Steps[0].Option("widths")!.StartsWith("1:", StringComparison.Ordinal)),
            "an option's range drawn with fractions");
        Check(written.Sample(9).Equals(written.Sample(9)), "the same seed, the same pipeline");

        var nested = layout.Render(LatinRows, TestPageStyle() with { Options = PageOptionsOf(("shapes", "box,grid=3x5,in=last")) }, 1);
        Check(nested.Blocks.Count(b => b.Kind == "region") == 15 && nested.Blocks.Count(b => b.Kind == "shape") == 2, $"in=last: the grid in the box's region takes its place: {nested.Description}");
        var bare = layout.Render(LatinRows, TestPageStyle() with { Options = PageOptionsOf(("shapes", "none")) }, 1);
        Check(bare.Text == string.Join("\n", LatinRows.Body) && !bare.Blocks.Any(b => b.Kind == "shape"), "no shapes: the lines in rows");

        // Pipelines as text and JSON; refusals.
        var pipeline = ShapePipeline.FromJson(JsonNode.Parse("""[{"name": "grid", "value": "2x3", "widths": "1:2:1"}, "line,dash=dotted"]"""));
        Check(pipeline.ToString() == "grid=2x3,widths=1:2:1,line,dash=dotted" && ShapePipeline.Parse(pipeline.ToString()).Equals(pipeline), $"JSON and text: {pipeline}");
        ThrowsNaming<ArgumentException>(() => ShapePipeline.Parse("nope"), "an unknown shape", "nope", "grid");
        ThrowsNaming<ArgumentException>(() => ShapePipeline.Parse("grid=3x4,nope=1"), "an unknown option", "nope", "widths");
        ThrowsNaming<ArgumentException>(() => ShapePipeline.Parse("grid=3x4,widths=a:b"), "a bad list", "widths");
        Throws<ArgumentException>(() => ShapePipeline.Parse("grid=zero"), "a grid without ROWSxCOLS");
        Throws<ArgumentException>(() => ShapePipeline.Parse("box,at=0.5:0.5"), "a box leaving its box");
        Throws<ArgumentException>(() => ShapePipeline.Parse("box,dash=wavy"), "an unknown dash");
        Throws<ArgumentException>(() => ShapePipeline.Parse("polyline"), "a polyline without points");
        Throws<ArgumentException>(() => ShapeProfiles.Resolve("grid@2"), "a chance above 1");
        Throws<ArgumentException>(() => ShapeProfiles.Resolve("grid=3x4,row-height=5..2"), "a range upside down");
        Throws<NotSupportedException>(() => ShapeProfiles.Resolve("nope"), "an unknown profile");
        Throws<NotSupportedException>(() => Pens.Get("nope"), "an unknown pen");
        ThrowsNaming<ArgumentException>(() => Pens.Get("ballpoint").Read(PageOptionsOf(("pen-wobble", "x"))), "a bad pen value", "pen-wobble");
        ThrowsNaming<ArgumentException>(() => Pens.Get("ballpoint").Read(PageOptionsOf(("wobble", "1"))), "a key the pen does not take", "pen-wobble");
        ThrowsNaming<ArgumentException>(() => TextBindings.Get("sequence").Read(PageOptionsOf(("take", "rest:rest"))), "rest twice", "take");
        Throws<ArgumentException>(() => layout.Render(LatinRows, TestPageStyle() with { Options = PageOptionsOf(("cell-align", "sideways")) }, 1), "an unknown alignment");

        Pens.Register(new LoudPen());
        try
        {
            Check(Pens.Origin(Pens.Ideal) != Overrides.Library && Pens.Default(Pens.Ideal) is { } library && library.Name == Pens.Ideal && Pens.Get("ballpoint").Name == "ballpoint",
                "the app's ideal pen over the library's, the ballpoint the library's");
            Throws<InvalidOperationException>(() => layout.Render(LatinRows, TestPageStyle() with { Options = PageOptionsOf(("shapes", "grid=1x1")) }, 1), "the app's pen draws");
        }
        finally
        {
            Pens.Unregister(Pens.Ideal);
        }

        Check(Pens.Origin(Pens.Ideal) == Overrides.Library, "unregistered: the library's ideal pen again");
    }

    // An app's "ideal" pen that refuses to draw: proves the page draws through the registered pen.
    internal sealed class LoudPen : IPen
    {
        public string Name => Pens.Ideal;

        public string Summary => "refuses to draw";

        public IReadOnlySet<string> Keys => OptionKeys.None;

        public PenSettings Read(IReadOnlyDictionary<string, string> options) => PenSettings.None;

        public float Stray(PenSettings settings, float width) => 0;

        public void Stroke(ReadOnlySpan<PenPoint> path, bool closed, PenSettings settings, int seed, PenStroke stroke) => throw new InvalidOperationException("the app's pen");
    }

    internal static void ShapeOptionsLeaveLetters(Device device)
    {
        _ = device;
        var options = PageOptionsOf(("shapes", "grid"), ("pen", "ballpoint"), ("bind", "sequence"), ("blank", "0.5"), ("blue-pen", "1"), ("cell-align", "end"));
        foreach (string name in new[] { "letter", "plain" })
        {
            foreach (var content in new[] { ArabicLetter, EnglishLetter })
            {
                foreach (bool color in new[] { true, false })
                {
                    for (int seed = 1; seed <= 3; seed++)
                    {
                        var plain = PageLayouts.Get(name).Render(content, TestPageStyle(color), seed);
                        var given = PageLayouts.Get(name).Render(content, TestPageStyle(color) with { Options = options }, seed);
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
            var plain = PageLayouts.Get("letter").Render(content, style, 2);
            var wobbly = PageLayouts.Get("letter").Render(content, style with { Options = PageOptionsOf(("pen-wobble", "3")) }, 2);
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
        string[] names = [.. Ornaments.Emblems.Split('|'), .. Ornaments.Stamps.Split('|'), Ornaments.Signature];
        Check(names.Length == 13 && names.All(ShapeProfiles.Names.Contains), $"the ornaments registered: {string.Join(", ", ShapeProfiles.Names)}");
        foreach (string name in names)
        {
            var profile = ShapeProfiles.Get(name);
            var steps = new Dictionary<string, List<ShapeStep>>();
            string? stray = null;
            int texts = 0;
            for (int seed = 0; seed < 12; seed++)
            {
                var box = new ShapeContext { X = 40, Y = 30, Width = 200, Height = 200, StrokeWidth = 4, RightToLeft = seed % 2 == 1, Seed = seed };
                var (drawn, _) = ShapesLayout.Plan(profile.Sample(seed), box, seed);
                foreach (var (step, plan) in drawn)
                {
                    (steps.TryGetValue(step.Name, out var list) ? list : steps[step.Name] = []).Add(step);
                    foreach (var path in plan.Paths)
                    {
                        var (x0, y0, x1, y1) = path.Bounds();
                        float reach = path.Width * box.StrokeWidth / 2 + 0.5f;
                        stray ??= x0 < box.X - reach || y0 < box.Y - reach || x1 > box.Right + reach || y1 > box.Bottom + reach ? $"seed {seed}: {step} draws {x0:F1},{y0:F1} to {x1:F1},{y1:F1}" : null;
                    }

                    foreach (var area in plan.Areas)
                    {
                        var (x0, y0, x1, y1) = area.Bounds();
                        stray ??= x0 < box.X - 0.5f || y0 < box.Y - 0.5f || x1 > box.Right + 0.5f || y1 > box.Bottom + 0.5f ? $"seed {seed}: {step} fills {x0:F1},{y0:F1} to {x1:F1},{y1:F1}" : null;
                    }

                    foreach (var text in plan.Texts)
                    {
                        var (x0, y0, x1, y1) = text.Bounds();
                        texts++;
                        stray ??= x0 < box.X - 0.5f || y0 < box.Y - 0.5f || x1 > box.Right + 0.5f || y1 > box.Bottom + 0.5f ? $"seed {seed}: {step}'s text may reach {x0:F1},{y0:F1} to {x1:F1},{y1:F1}" : null;
                    }
                }
            }

            Check(stray is null && steps.Count > 0, $"{name}: inside its box ({stray})");
            Check(name.StartsWith("emblem-rings", StringComparison.Ordinal) || name.StartsWith("emblem-star", StringComparison.Ordinal) || name == "scribble" || texts > 0, $"{name}: says something ({texts} texts)");
            // The suite places each step itself (its random cases add at and size), so it gets them without the profile's own.
            foreach (var (shape, given) in steps)
            {
                var unplaced = given.Select(s => new ShapeStep(s.Name, s.Value, s.Options.Where(o => o.Key is not ("at" or "size")))).Distinct().Take(6).ToList();
                Conformance.CheckShape(shape, steps: unplaced).ThrowIfFailed();
            }
        }

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

    internal static void ShapeTextsWritten(Device device)
    {
        CpuOnly(device);
        var face = SubsetFace();
        var lettering = new ShapeLettering(new TextStyle(face, 20), "Northwind Trading Company", RightToLeft: false);
        var pen = Pens.Get(Pens.Ideal);
        var settings = pen.Read(PageOptions.None);
        const int width = 280, height = 260;
        var style = new DocumentShapeStyle { Scale = 0.75f, StrokeWidth = 4 };
        var problems = new List<string>();
        int said = 0;
        double worst = 0;
        foreach (string name in (string[])[.. Ornaments.Emblems.Split('|'), .. Ornaments.Stamps.Split('|')])
        {
            var profile = ShapeProfiles.Get(name);
            for (int seed = 0; seed < 3; seed++)
            {
                var box = new ShapeContext { X = 40, Y = 30, Width = 200, Height = 200, StrokeWidth = 4, Seed = seed };
                var (drawn, _) = ShapesLayout.Plan(profile.Sample(seed), box, seed);
                var plans = seed == 2 ? ShapeDrawing.Turned(drawn, 140, 130, 0.2f, 0, 0) : drawn;    // turned, as a stamp is
                var expected = plans.SelectMany(d => d.Plan.Texts).Select(t => ShapeDrawing.Say(t, lettering)).OfType<string>().ToList();
                said += expected.Count;
                using var output = new MemoryStream();
                using (var builder = DocumentWriters.Create(output, "pdf"))
                {
                    var page = builder.AddPage(width * 0.75f, height * 0.75f);
                    foreach (var (_, plan) in plans)
                    {
                        DocumentShapes.Write(page, plan, style, lettering.Style, lettering.PageText, lettering.RightToLeft);
                    }

                    builder.Finish();
                }

                using var document = DocumentFormats.Open(output.ToArray(), name + ".pdf", new DocumentOptions { Layout = DocumentTextLayouts.Content });
                var read = document.ReadPage(0, DocumentParts.All);
                string text = string.Concat(read.Runs.Select(r => r.Text));
                if (text != string.Concat(expected))
                {
                    problems.Add($"{name} seed {seed}: read '{text}', the plan says '{string.Concat(expected)}'");
                }

                // The same plans drawn on a page with the ideal pen, against the written page drawn back.
                using var canvas = new PageCanvas(width, height);
                ShapeDrawing.Draw(canvas, canvas.Ink, plans, pen, settings, 0, 4, null, lettering);
                var image = canvas.Compose((1, 1, 1), 1, (0, 0, 1), true);
                var back = DocumentPageDrawers.DrawPage(document, 0, new DocumentDrawOptions { Dpi = 96 }).Image;
                var (share, _) = PdfDrawDifference(image, back, 32);
                worst = Math.Max(worst, share);
                if (back.Width != width || back.Height != height || share > 0.01)
                {
                    problems.Add($"{name} seed {seed}: drawn back {back.Width} x {back.Height}, {share:P2} of the pixels off by more than 32 levels");
                }
            }
        }

        Check(problems.Count == 0 && said > 20, $"plan texts written ({said} texts, the worst page {worst:P2} off):\n  " + string.Join("\n  ", problems));
    }

    internal static void LetterOrnamentPages(Device device)
    {
        _ = device;
        var letter = PageLayouts.Get("letter");
        string[] emblems = Ornaments.Emblems.Split('|'), stamps = Ornaments.Stamps.Split('|');
        foreach (var content in new[] { ArabicLetter, EnglishLetter })
        {
            var style = TestPageStyle() with { Emblem = 1, Stamp = 1 };
            for (int i = 0; i < emblems.Length; i++)
            {
                int seed = 3 + i;
                string emblem = emblems[i], stamp = stamps[i % stamps.Length];
                var page = letter.Render(content, style with { Options = PageOptionsOf(("emblem-shapes", emblem), ("stamp-shapes", stamp)) }, seed);
                string where = $"{content.Language}, {emblem}, {stamp}, seed {seed}";
                Check(page.Text == content.ReadingOrder() && page.Description.Contains($", emblem {emblem},", StringComparison.Ordinal)
                      && page.Description.Contains($", stamp {stamp}", StringComparison.Ordinal) && page.Description.Contains(", signed scribble with ballpoint", StringComparison.Ordinal),
                    $"{where}: the text unchanged, the ornaments named ({page.Description})");
                Check(new[] { "emblem", "stamp", "scribble" }.All(k => page.Blocks.Count(b => b.Kind == k && b.Text.Length == 0 && b.Width > 0 && b.Height > 0
                    && b.X >= 0 && b.Y >= 0 && b.X + b.Width <= page.Image.Width && b.Y + b.Height <= page.Image.Height) == 1), $"{where}: a block each, inside the page");
                Check(SamePage(page, letter.Render(content, style with { Options = PageOptionsOf(("emblem-shapes", emblem), ("stamp-shapes", stamp)) }, seed)),
                    $"{where}: the same seed, the same page");

                // Another emblem and stamp on the same seed: the pages differ inside the two ornaments' blocks and nowhere else.
                string otherEmblem = emblems[(i + 1) % emblems.Length], otherStamp = stamps[(i + 1) % stamps.Length];
                var other = letter.Render(content, style with { Options = PageOptionsOf(("emblem-shapes", otherEmblem), ("stamp-shapes", otherStamp)) }, seed);
                var ornaments = page.Blocks.Concat(other.Blocks).Where(b => b.Kind is "emblem" or "stamp").ToList();
                var changed = DifferingPixels(page, other).ToList();
                int outside = changed.Count(p => !ornaments.Any(b => Within(p, b, 1)));
                Check(outside == 0 && changed.Count > 0 && page.Blocks.Where(b => b.Text.Length > 0).SequenceEqual(other.Blocks.Where(b => b.Text.Length > 0)),
                    $"{where}: against {otherEmblem} and {otherStamp}, {changed.Count} pixels changed, {outside} of them away from the ornaments; the text's parts where they were");
            }
        }

        // The signature: drawn by the pen asked, only inside its block; blue at blue-signature 1, the name black.
        var plainStyle = TestPageStyle() with { Emblem = 0, Stamp = 0 };
        var ballpoint = letter.Render(EnglishLetter, plainStyle with { BlueSignature = 1 }, 5);
        var ideal = letter.Render(EnglishLetter, plainStyle with { BlueSignature = 1, Options = PageOptionsOf(("signature-pen", "ideal")) }, 5);
        var scribbles = ballpoint.Blocks.Concat(ideal.Blocks).Where(b => b.Kind == "scribble").ToList();
        int away = DifferingPixels(ballpoint, ideal).Count(p => !scribbles.Any(b => Within(p, b, 1)));
        Check(scribbles.Count == 2 && away == 0 && !SamePage(ballpoint, ideal) && ideal.Description.Contains(", signed scribble with ideal", StringComparison.Ordinal)
              && ballpoint.Text == EnglishLetter.ReadingOrder(), $"the signature drawn by the pen asked, {away} pixels changed away from it ({ideal.Description})");
        static (int R, int B) DarkestInk(RenderedPage page, PageBlock block)
        {
            var planes = Planes(page.Image);
            int width = page.Image.Width, best = int.MaxValue, r = 0, b = 0;
            for (int y = (int)block.Y; y < (int)(block.Y + block.Height); y++)
            {
                for (int x = (int)block.X; x < (int)(block.X + block.Width); x++)
                {
                    int i = y * width + x, sum = planes[0][i] + planes[1][i] + planes[2][i];
                    (best, r, b) = sum < best ? (sum, planes[0][i], planes[2][i]) : (best, r, b);
                }
            }

            return (r, b);
        }

        // The signature's part above the name (the two may touch), and the name's block.
        static PageBlock Above(RenderedPage page)
        {
            var (signature, nameBlock) = (page.Blocks.Single(b => b.Kind == "scribble"), page.Blocks.Single(b => b.Kind == "signature"));
            return signature with { Height = Math.Max(1, nameBlock.Y - signature.Y) };
        }

        var blueInk = DarkestInk(ballpoint, Above(ballpoint));
        var name = DarkestInk(ballpoint, ballpoint.Blocks.Single(b => b.Kind == "signature"));
        var black = letter.Render(EnglishLetter, plainStyle with { BlueSignature = 0 }, 5);
        var blackInk = DarkestInk(black, Above(black));
        Check(blueInk.B - blueInk.R > 40 && Math.Abs(name.B - name.R) < 30 && Math.Abs(blackInk.B - blackInk.R) < 30 && ballpoint.Description.Contains(", blue signature", StringComparison.Ordinal),
            $"a blue signature over a black name (signature {blueInk}, name {name}), a black one at blue-signature 0 ({blackInk})");

        var unsigned = letter.Render(EnglishLetter, plainStyle with { Options = PageOptionsOf(("signature", "0")) }, 5);
        var signedName = black.Blocks.Single(b => b.Kind == "signature");
        var bareName = unsigned.Blocks.Single(b => b.Kind == "signature");
        Check(!unsigned.Blocks.Any(b => b.Kind == "scribble") && !unsigned.Description.Contains("signed", StringComparison.Ordinal) && bareName.Y < signedName.Y && unsigned.Text == black.Text,
            $"signature=0: no signature drawn, the name where the signature would stand ({bareName.Y:F0} above {signedName.Y:F0})");

        // Their settings leave a page without them as it was: no emblem, stamp or signature drawn, the same bytes.
        var bare = plainStyle with { Options = PageOptionsOf(("signature", "0")) };
        var given = plainStyle with
        {
            Options = PageOptionsOf(("signature", "0"), ("emblem-shapes", "emblem-star"), ("stamp-shapes", "stamp-oval|stamp-ring"), ("stamp-turn", "30"), ("stamp-ink", "1"),
                ("stamp-black", "1"), ("signature-pen", "ideal"), ("emblem-pen", "ballpoint"), ("stamp-pen", "ballpoint"), ("signature-shapes", "scribble,letters=3")),
        };
        foreach (var content in new[] { ArabicLetter, EnglishLetter, Newsletter })
        {
            for (int seed = 1; seed <= 3; seed++)
            {
                var a = letter.Render(content, bare, seed);
                var b = letter.Render(content, given, seed);
                Check(SamePage(a, b) && a.Description == b.Description, $"{content.Language}, seed {seed}: the ornaments' settings leave a page without them as it was");
            }

            // A page with no signature block draws none, whatever the chance: the same bytes as at chance 0.
            if (content.Signature.Count == 0)
            {
                Check(SamePage(letter.Render(content, plainStyle, 4), letter.Render(content, bare, 4)), "no signature block: nothing signed, the page as at signature=0");
            }
        }

        // Bad settings, refused naming them.
        foreach (var (key, value) in new[] { ("stamp-turn", "abc"), ("stamp-turn", "200"), ("stamp-ink", "2"), ("stamp-black", "1.5"), ("signature", "-1") })
        {
            ThrowsNaming<ArgumentException>(() => letter.Render(EnglishLetter, TestPageStyle() with { Options = PageOptionsOf((key, value)) }, 1), $"{key}={value}", key);
        }

        Throws<ArgumentException>(() => letter.Render(EnglishLetter, TestPageStyle() with { Options = PageOptionsOf(("emblem-shapes", "emblem-star||emblem-box")) }, 1), "an empty profile between |");
        Throws<NotSupportedException>(() => letter.Render(EnglishLetter, TestPageStyle() with { Options = PageOptionsOf(("stamp-shapes", "nope")) }, 1), "an unknown stamp profile");
        Throws<NotSupportedException>(() => letter.Render(EnglishLetter, TestPageStyle() with { Options = PageOptionsOf(("signature-pen", "nope")) }, 1), "an unknown pen");
    }
}
