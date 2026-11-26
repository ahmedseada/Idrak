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
        ("shapes: a text says one of its words the page's fonts draw (those in the page's direction first), the page's own text or its initials when it has none, nothing when none can be drawn", OrnamentProfiles),
    ];

    internal static IReadOnlyDictionary<string, string> PageOptionsOf(params (string Key, string Value)[] options) =>
        options.ToDictionary(o => o.Key, o => o.Value, StringComparer.OrdinalIgnoreCase);

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
