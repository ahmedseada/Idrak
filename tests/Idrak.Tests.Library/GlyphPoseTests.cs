// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Data;
using Idrak.Abstraction.Testing;
using Idrak.Imaging;
using Idrak.Text;

// Each glyph drawn a little differently, as a pen writes, in every layout, through one contract (IGlyphPose,
// GlyphPoses): "none" draws text as set, "drift" moves it within declared bounds, a cluster at a time.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] GlyphPoseGroup =
    [
        ("glyph poses: the suite passes on none (the identity) and drift (its defaults and wider settings)", GlyphPoseSuites),
        ("glyph poses: none leaves letter, plain and shapes pages byte for byte as they were; drift draws the same page from the same seed, the text unchanged, its ink only where the text was within the pose's bounds, a region's text inside its region", GlyphPosePages),
        ("glyph poses: marks stay on their base (one pose a cluster), and a joined Arabic word drawn with drift has as many ink pieces as drawn as set", GlyphPoseJoins),
        ("glyph poses: page settings read pose and its keys into the options and refuse an unknown pose, a bad value or an unknown key, naming the valid ones", GlyphPoseSettingsRead),
    ];

    internal static readonly IReadOnlyDictionary<string, string> WideDrift = PageOptionsOf(
        ("pose-slant", "-0.2..0.3"), ("pose-size", "0.1"), ("pose-baseline", "0.1..0.2"), ("pose-rotate", "3"), ("pose-pressure", "0.5"));

    internal static void GlyphPoseSuites(Device device)
    {
        _ = device;
        Check(GlyphPoses.Names.Contains(GlyphPoses.None) && GlyphPoses.Names.Contains("drift"), $"poses: {string.Join(", ", GlyphPoses.Names)}");
        Conformance.CheckGlyphPose(GlyphPoses.None, identity: true).ThrowIfFailed();
        Conformance.CheckGlyphPose("drift", settings: [WideDrift, PageOptionsOf(("pose-slant", "0.1"), ("pose-pressure", "0"))]).ThrowIfFailed();
    }

    internal static void GlyphPosePages(Device device)
    {
        _ = device;
        var style = TestPageStyle();
        foreach (var (name, content, options) in new[]
        {
            ("letter", ArabicLetter, PageOptions.None), ("plain", EnglishLetter, PageOptions.None),
            ("shapes", LatinRows, PageOptionsOf(("shapes", "grid=3x5"), ("pen", "ballpoint"))),
        })
        {
            var layout = PageLayouts.Get(name);
            for (int seed = 1; seed <= 2; seed++)
            {
                var plain = layout.Render(content, style with { Options = options }, seed);
                var none = layout.Render(content, style with { Options = WithOptions(options, ("pose", "none")) }, seed);
                Check(SamePage(plain, none) && plain.Description == none.Description, $"{name}, seed {seed}: pose=none draws the page as without a pose");

                var drifted = WithOptions(options, ("pose", "drift"));
                var drift = layout.Render(content, style with { Options = drifted }, seed);
                Check(SamePage(drift, layout.Render(content, style with { Options = drifted }, seed)), $"{name}, seed {seed}: drift, the same seed the same page");
                Check(drift.Text == plain.Text && drift.Description.Contains("pose drift", StringComparison.Ordinal), $"{name}, seed {seed}: the text unchanged ({drift.Description})");
                Check(!SamePage(drift, plain), $"{name}, seed {seed}: drift moves the glyphs");

                // Where the posed page differs from the page as set: only near the text, within what the pose may move it.
                var bounds = GlyphPoses.Get("drift").Bounds(GlyphPoses.Get("drift").Read(PageOptions.None));
                float pad = bounds.Reach(60, 90) + 2;
                var text = plain.Blocks.Where(b => b.Text.Length > 0).ToList();
                int outside = DifferingPixels(drift, plain).Count(p => !text.Any(b => p.X >= b.X - pad && p.X <= b.X + b.Width + pad && p.Y >= b.Y - pad && p.Y <= b.Y + b.Height + pad));
                Check(outside == 0, $"{name}, seed {seed}: {outside} pixels changed away from the text");
                if (name == "shapes")
                {
                    var blank = layout.Render(content, style with { Options = WithOptions(drifted, ("blank", "1")) }, seed);
                    var regions = drift.Blocks.Where(b => b.Kind == "region").ToList();
                    var (away, inked) = TextInk(drift, blank, regions);
                    Check(away == 0 && inked.All(i => i), $"shapes, seed {seed}: posed text ink inside its regions ({away} pixels outside)");
                }
            }
        }
    }

    internal static IReadOnlyDictionary<string, string> WithOptions(IReadOnlyDictionary<string, string> options, params (string Key, string Value)[] more)
    {
        var all = new Dictionary<string, string>(options, StringComparer.OrdinalIgnoreCase);
        foreach (var (key, value) in more)
        {
            all[key] = value;
        }

        return all;
    }

    internal static bool SamePage(RenderedPage a, RenderedPage b) =>
        Planes(a.Image).Zip(Planes(b.Image)).All(p => p.First.SequenceEqual(p.Second)) && a.Image.Channels == b.Image.Channels && a.Text == b.Text && a.Blocks.SequenceEqual(b.Blocks);

    // The pixels where two pages of one size differ.
    internal static IEnumerable<(int X, int Y)> DifferingPixels(RenderedPage a, RenderedPage b)
    {
        var (pa, pb) = (Planes(a.Image), Planes(b.Image));
        int width = a.Image.Width;
        for (int i = 0; i < pa[0].Length; i++)
        {
            for (int c = 0; c < pa.Length; c++)
            {
                if (pa[c][i] != pb[c][i])
                {
                    yield return (i % width, i / width);
                    break;
                }
            }
        }
    }

    internal static void GlyphPoseJoins(Device device)
    {
        _ = device;
        var face = SubsetFace();
        var drift = GlyphPoses.Get("drift");

        // Marks: every glyph of a cluster gets its cluster's pose, and a mark's ink centre stays over its base's ink.
        string? marked = new[] { "بَدِيعٌ", "été", "äö" }.FirstOrDefault(t => t.All(ch => face.Glyph(ch) != 0));
        Check(marked is not null, "the test font has a base and mark to try");
        var block = TextLayout.Layout(marked!, new TextStyle(face, 80), float.PositiveInfinity, TextAlignment.Start);
        int clusters = 0;
        for (int seed = 1; seed <= 6; seed++)
        {
            var painter = new GlyphPainter(drift, drift.Read(WideDrift), seed);
            foreach (var group in painter.Posed(block, 0).GroupBy(p => p.Glyph.Cluster).Where(g => g.Count() > 1))
            {
                clusters++;
                Check(group.Select(p => p.Pose).Distinct().Count() == 1, $"seed {seed}: cluster {group.Key} has one pose");
                var boxes = group.Select(p => (p.Glyph, Box: PosedInkBox(painter, p.Glyph, p.Pose))).Where(p => p.Box.Area > 0).OrderByDescending(p => p.Box.Area).ToList();
                var bases = boxes[0].Box;
                foreach (var (_, box) in boxes.Skip(1))
                {
                    float centre = (box.X0 + box.X1) / 2;
                    Check(centre >= bases.X0 && centre <= bases.X1, $"seed {seed}: a mark's centre {centre:F1} over its base's ink {bases.X0:F1}..{bases.X1:F1}");
                }
            }
        }

        Check(clusters > 0, "the text has clusters of a base and a mark");

        // A joined word: as many ink pieces drifted as set.
        var word = TextLayout.Layout("المستقبل", new TextStyle(face, 90), float.PositiveInfinity, TextAlignment.Start, rightToLeft: true);
        int set = InkPieces(word, new GlyphPainter(GlyphPoses.Get(GlyphPoses.None), GlyphPoseSettings.None, 1));
        for (int seed = 1; seed <= 8; seed++)
        {
            int drifted = InkPieces(word, new GlyphPainter(drift, drift.Read(PageOptions.None), seed));
            Check(drifted == set, $"seed {seed}: the joined word in {drifted} ink pieces drifted, {set} as set");
        }
    }

    // A glyph's posed ink box, in pixels relative to the block.
    internal static (float X0, float X1, float Area) PosedInkBox(GlyphPainter painter, PlacedGlyph g, GlyphPose pose)
    {
        var outline = g.Font.Outline(g.Glyph);
        if (outline.IsEmpty)
        {
            return (0, 0, 0);
        }

        var posed = painter.Transform(outline, g.Scale, pose, (float)PortableMath.Cos(pose.Rotation), (float)PortableMath.Sin(pose.Rotation));
        var (minX, minY, maxX, maxY) = posed.Bounds();
        return (g.X + minX, g.X + maxX, (maxX - minX) * (maxY - minY));
    }

    // The 8-connected pieces of ink (coverage above a half) of a block drawn with a painter.
    internal static int InkPieces(TextBlock block, GlyphPainter painter)
    {
        int width = (int)Math.Ceiling(block.Width) + 80, height = (int)Math.Ceiling(block.Height) + 80;
        var layer = new float[width * height];
        painter.Draw(block, 40, 40, layer, width, height);
        var seen = new bool[layer.Length];
        var stack = new Stack<int>();
        int pieces = 0;
        for (int start = 0; start < layer.Length; start++)
        {
            if (seen[start] || layer[start] <= 0.5f)
            {
                continue;
            }

            pieces++;
            seen[start] = true;
            stack.Push(start);
            while (stack.Count > 0)
            {
                int at = stack.Pop(), x = at % width, y = at / width;
                for (int dy = -1; dy <= 1; dy++)
                {
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        int nx = x + dx, ny = y + dy, n = ny * width + nx;
                        if ((uint)nx < (uint)width && (uint)ny < (uint)height && !seen[n] && layer[n] > 0.5f)
                        {
                            seen[n] = true;
                            stack.Push(n);
                        }
                    }
                }
            }
        }

        return pieces;
    }

    internal static void GlyphPoseSettingsRead(Device device)
    {
        _ = device;
        var style = new PageStyle { Fonts = [SubsetFace()] };
        var read = SyntheticPages.Style(style, "pose=drift, pose-slant=-0.1..0.1, pose-size=0.03, pose-baseline=0..0.05, pose-rotate=0.5..2, pose-pressure=0.1");
        Check(read.Options.Count == 6 && read.Options["pose"] == "drift" && read.Options["pose-rotate"] == "0.5..2", $"the pose keys in the options: {string.Join(", ", read.Options)}");
        Check(SyntheticPages.StyleKeys.Contains("pose") && new[] { "pose-slant", "pose-size", "pose-baseline", "pose-rotate", "pose-pressure" }.All(SyntheticPages.KindStyleKeys().Contains),
            $"the keys: {string.Join(", ", SyntheticPages.KindStyleKeys())}");
        ThrowsNaming<ArgumentException>(() => SyntheticPages.Style(style, "pose-wobble=1"), "an unknown key", "pose-slant", "pose", "pen-wobble");
        foreach (string bad in new[] { "pose=nope", "pose-slant=2", "pose-size=x", "pose-baseline=0.9", "pose-rotate=-1", "pose-pressure=1.5" })
        {
            Throws<ArgumentException>(() => SyntheticPages.Style(style, bad), bad);
        }

        ThrowsNaming<ArgumentException>(() => GlyphPoses.Get("drift").Read(PageOptionsOf(("slant", "0.1"))), "a key drift does not take", "pose-slant");
        Throws<ArgumentException>(() => GlyphPoses.Get(GlyphPoses.None).Read(PageOptionsOf(("pose-slant", "0.1"))), "none takes no keys");
        Throws<NotSupportedException>(() => PageLayouts.Get("letter").Render(ArabicLetter, TestPageStyle() with { Options = PageOptionsOf(("pose", "nope")) }, 1), "an unknown pose on a page");
    }
}
