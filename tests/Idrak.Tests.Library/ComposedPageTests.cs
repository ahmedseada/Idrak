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
    ];

    internal const string Stacked = "text,lines=2,flow=down,grid=auto x 5,flow=down,text,lines=1,flow=down";

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
