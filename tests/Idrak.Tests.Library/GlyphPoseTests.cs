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
        ("glyph poses: marks stay on their base (one pose a cluster), and a joined Arabic word drawn with drift has as many ink pieces as drawn as set", GlyphPoseJoins),
    ];

    internal static readonly IReadOnlyDictionary<string, string> WideDrift = PageOptionsOf(
        ("pose-slant", "-0.2..0.3"), ("pose-size", "0.1"), ("pose-baseline", "0.1..0.2"), ("pose-rotate", "3"), ("pose-pressure", "0.5"));

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
        var drift = new DriftPose();

        // Marks: every glyph of a cluster gets its cluster's pose, and a mark's ink centre stays over its base's ink.
        string? marked = new[] { "بَدِيعٌ", "été", "äö" }.FirstOrDefault(t => t.All(ch => face.Glyph(ch) != 0));
        Check(marked is not null, "the test font has a base and mark to try");
        var block = TextLayout.Layout(marked!, new TextStyle(face, 80), float.PositiveInfinity, TextAlignment.Start);
        int clusters = 0;
        for (int seed = 1; seed <= 6; seed++)
        {
            var painter = new GlyphPainter(drift, "drift", drift.Read(WideDrift), seed);
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
        int set = InkPieces(word, new GlyphPainter(new NonePose(), GlyphPoses.None, GlyphPoseSettings.None, 1));
        for (int seed = 1; seed <= 8; seed++)
        {
            int drifted = InkPieces(word, new GlyphPainter(drift, "drift", drift.Read(PageOptions.None), seed));
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
}
