// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Frozen;

namespace Idrak.Imaging;

/// <summary>
/// The letter's ornaments as shape profiles: the emblems, the stamps and the signature the library registers
/// (<c>ShapeProfiles</c>), and the lists a letter draws one of by default (its <c>emblem-shapes</c>,
/// <c>stamp-shapes</c> and <c>signature-shapes</c> options). Every design is generic: rings, a seal, a star, a shield, a
/// monogram, boxes; their texts are the page's own name line or the generic office words below.
/// </summary>
internal static class Ornaments
{
    /// <summary>The emblems a letter draws one of unless its <c>emblem-shapes</c> option names others.</summary>
    public const string Emblems = "emblem-rings|emblem-seal|emblem-star|emblem-shield|emblem-badge|emblem-monogram|emblem-box";

    /// <summary>The stamps a letter draws one of unless its <c>stamp-shapes</c> option names others.</summary>
    public const string Stamps = "stamp-ring|stamp-seal|stamp-rectangle|stamp-oval|stamp-double";

    /// <summary>The signature a letter draws over its signature block's name unless its <c>signature-shapes</c> option names another.</summary>
    public const string Signature = "scribble";

    // Generic office stamp words in two directions: a stamp says one its page's fonts draw, in the page's direction.
    private const string StampWords = "words=RECEIVED:APPROVED:COPY:DRAFT:ORIGINAL:CHECKED:FILED:PAID:VERIFIED:URGENT:وارد:صادر:نسخة:معتمد:مسودة:أصل:عاجل:مدفوع";

    // Each drawn in a square box (an emblem's or a stamp's); the stroke a share of its size.
    private static readonly (string Name, string Text, string Summary)[] Profiles =
    [
        ("emblem-rings", "circle,width=1.6,circle,margin=0.15", "an emblem: a ring with an inner ring"),
        ("emblem-seal", "circle,width=1.5,ring-text,radius=0.94,font-size=0.17,circle,margin=0.2,width=0.8,lettering,initials=on,margin=0.3,font-size=0.7",
            "an emblem: a ring with the page's name round its top, an inner ring, the name's initials inside"),
        ("emblem-star", "circle,width=1.5,star=5..8,margin=0.12,inner=0.38..0.5,circle@0.5,margin=0.06,width=0.6", "an emblem: a ring with a star of 5 to 8 points inside"),
        ("emblem-shield", "shield,at=0.1:0,size=0.8:1,width=1.5,shield,at=0.1:0,size=0.8:1,margin=0.07,width=0.7,lettering,initials=on,at=0.25:0.12,size=0.5:0.4,font-size=0.8",
            "an emblem: a shield with an inner line, the page's initials in its upper part"),
        ("emblem-badge", "shield,at=0.1:0,size=0.8:1,width=1.5,hatch,at=0.18:0.07,size=0.64:0.3,spacing=3.5,width=0.6,lettering,initials=on,at=0.25:0.42,size=0.5:0.3,font-size=0.8",
            "an emblem: a shield, its upper part hatched, the page's initials below"),
        ("emblem-monogram", "circle,width=1.4,circle@0.5,margin=0.07,width=0.6,lettering,initials=on,margin=0.2,font-size=0.8", "an emblem: the page's initials in a circle"),
        ("emblem-box", "box,at=0:0.25,size=1:0.5,radius=0.15,width=1.5,box@0.6,at=0:0.25,size=1:0.5,margin=0.08,radius=0.12,width=0.6,lettering,at=0:0.25,size=1:0.5,margin=0.16,font-size=0.7",
            "an emblem: a rounded box (a second rule inside at a chance) with the page's name"),
        ("stamp-ring", $"circle,width=1.3,circle,margin=0.12,width=0.7,lettering,margin=0.22,font-size=0.5,{StampWords}", "a stamp: two rings with a generic word inside"),
        ("stamp-seal", $"circle,width=1.3,ring-text,radius=0.95,font-size=0.15,circle,margin=0.21,width=0.7,lettering,margin=0.3,font-size=0.5,{StampWords}",
            "a stamp: a seal, the page's name round its top, a generic word inside"),
        ("stamp-rectangle", $"box,at=0:0.25,size=1:0.5,radius=0.06,width=1.3,lettering,at=0:0.25,size=1:0.5,margin=0.12,font-size=0.7,{StampWords}", "a stamp: a rectangle with a generic word"),
        ("stamp-oval", $"ellipse,at=0:0.18,size=1:0.64,width=1.3,ellipse,at=0:0.18,size=1:0.64,margin=0.07,width=0.6,lettering,at=0.14:0.34,size=0.72:0.32,font-size=0.8,{StampWords}",
            "a stamp: an oval with an inner line and a generic word"),
        ("stamp-double", $"box,at=0:0.2,size=1:0.6,width=1.3,box,at=0:0.2,size=1:0.6,margin=0.05,width=0.6,lettering,at=0:0.24,size=1:0.28,margin=0.08,font-size=0.8,{StampWords},lettering,at=0:0.5,size=1:0.26,margin=0.08,font-size=0.7",
            "a stamp: a box of two rules, a generic word over the page's name"),
        ("scribble", "scribble", "a signature: a pen's scribble of letter-like loops, never text"),
    ];
}

/// <summary>What the text shapes share: their words, pick and initials read from a step.</summary>
internal static class ShapeWords
{
    /// <summary>The step's words (<c>words=A:B:C</c>; none: the page's own text), the pick from its seed, and whether it says initials.</summary>
    public static (IReadOnlyList<string> Words, int Pick, bool Initials) Read(ShapeStep step, ShapeContext context) =>
        (step.List("words") ?? [], new Random(context.Seed).Next(), step.Switch("initials", false));

    /// <summary>Reads the options every text shape takes, throwing on a bad one.</summary>
    public static void Check(ShapeStep step)
    {
        _ = step.List("words");
        _ = step.Switch("initials", false);
    }
}

/// <summary>"ring-text": a text round a ring in the middle of its box, along its top or its bottom, read left to right.</summary>
internal sealed class RingTextShape : IShape
{
    public string Name => "ring-text";

    public string Summary => "a text round a ring in the middle of its box, its ink inside the ring's edge: radius (the edge, a share of half the smaller side; 1), font-size (a share of "
        + "the radius; 0.16), side (top or bottom; top: the glyphs' tops outward, bottom: toward the middle), span (the most degrees it takes; 150), words (A:B:..., one drawn; "
        + "none: the page's own text), initials (on: the first letters of the text's first three words)";

    public void Check(ShapeStep step)
    {
        ShapeGeometry.NoValue(step);
        _ = step.Number("radius", 1, 0.05f, 1);
        _ = step.Number("font-size", 0.16f, 0.02f, 0.45f);
        _ = step.Choice("side", "top", "top", "bottom");
        _ = step.Number("span", 150, 10, 300);
        ShapeWords.Check(step);
    }

    public ShapePlan Plan(ShapeStep step, ShapeContext context)
    {
        float edge = step.Number("radius", 1, 0.05f, 1) * Math.Min(context.Width, context.Height) / 2;
        float size = step.Number("font-size", 0.16f, 0.02f, 0.45f) * edge;
        if (!(size > 0.01f))
        {
            return ShapePlan.Empty;
        }

        // The baseline a size inside the edge: the ink, within a size of it on either side, stays inside the edge.
        float r = edge - size, cx = context.X + context.Width / 2, cy = context.Y + context.Height / 2;
        double span = step.Number("span", 150, 10, 300);
        bool top = step.Choice("side", "top", "top", "bottom") == "top";
        // Left to right either way (the glyphs are in the order they are drawn): over the top clockwise, along the bottom counterclockwise.
        var path = top
            ? ShapeGeometry.Arc(context with { RightToLeft = false }, cx, cy, r, r, 90 + span / 2, 90 - span / 2, closed: false)
            : ShapeGeometry.Arc(context with { RightToLeft = false }, cx, cy, r, r, 270 - span / 2, 270 + span / 2, closed: false);
        var (words, pick, initials) = ShapeWords.Read(step, context);
        return new ShapePlan([]) { Texts = [new ShapeText(path, size, words, pick, initials, step.Tone())] };
    }
}

/// <summary>"lettering": a line of text set upright across the middle of its box.</summary>
internal sealed class LetteringShape : IShape
{
    public string Name => "lettering";

    public string Summary => "a line of text upright across the middle of its box, smaller when it is longer than the box is wide: font-size (a share of the box's height; 0.5), "
        + "words (A:B:..., one drawn; none: the page's own text), initials (on: the first letters of the text's first three words)";

    public void Check(ShapeStep step)
    {
        ShapeGeometry.NoValue(step);
        _ = step.Number("font-size", 0.5f, 0.02f, 1);
        ShapeWords.Check(step);
    }

    public ShapePlan Plan(ShapeStep step, ShapeContext context)
    {
        // The baseline under the middle so that capitals sit about it; its ink within a size of it stays in the box.
        float size = Math.Min(step.Number("font-size", 0.5f, 0.02f, 1) * context.Height, context.Height / 2.6f);
        if (!(size > 0.01f) || !(context.Width > 0))
        {
            return ShapePlan.Empty;
        }

        float y = context.Y + context.Height / 2 + 0.3f * size;
        var (words, pick, initials) = ShapeWords.Read(step, context);
        return new ShapePlan([]) { Texts = [new ShapeText([context.X, y, context.Right, y], size, words, pick, initials, step.Tone())] };
    }
}

/// <summary>"star": a star of N points in its box, one region inside.</summary>
internal sealed class StarShape : IShape
{
    // The value ("") is drawn whole by a profile: a count of points.
    public string Name => "star";

    public string Summary => "a star of N points in its box (star=N; 5), its first point at the top, a region inside: inner (its inner corners, a share of the outer; 0.45), pad (text sizes; 0.15)";

    public void Check(ShapeStep step)
    {
        _ = step.Integer("", 5, 3, 64);
        _ = step.Number("inner", 0.45f, 0.05f, 0.95f);
        _ = step.Number("pad", 0.15f, 0, 20);
    }

    public ShapePlan Plan(ShapeStep step, ShapeContext context)
    {
        int count = step.Integer("", 5, 3, 64);
        float inner = step.Number("inner", 0.45f, 0.05f, 0.95f);
        float cx = context.X + context.Width / 2, cy = context.Y + context.Height / 2, rx = context.Width / 2, ry = context.Height / 2;
        var points = new float[4 * count];
        for (int k = 0; k < 2 * count; k++)
        {
            double a = Math.PI / 2 + Math.PI * k / count;
            float share = k % 2 == 0 ? 1 : inner;
            points[2 * k] = cx + (context.RightToLeft ? -1 : 1) * share * rx * (float)PortableMath.Cos(a);
            points[2 * k + 1] = cy - share * ry * (float)PortableMath.Sin(a);
        }

        // The square inside the circle through the inner corners' sides, scaled to its box, less the stroke and the padding.
        float side = inner * (float)PortableMath.Cos(Math.PI / count) / MathF.Sqrt(2);
        float inset = ShapeGeometry.Half(step, context) + ShapeGeometry.Pad(step, context, 0.15f);
        var region = ShapeGeometry.Region(0, 0, cx - rx * side, cy - ry * side, cx + rx * side, cy + ry * side, inset, inset, inset, inset) with { Paragraphs = step.HoldsParagraphs() };
        return new ShapePlan([ShapeGeometry.Path(step, points, closed: true)], null, [region]);
    }
}

/// <summary>"shield": a shield's outline in its box (a flat top, straight sides, curving to a point at the bottom), one region in its upper part.</summary>
internal sealed class ShieldShape : IShape
{
    public string Name => "shield";

    public string Summary => "a shield in its box: a flat top, straight sides down to the shoulder, then curving to a point at the bottom's middle, a region above the shoulder: "
        + "shoulder (a share of the height; 0.5), pad (text sizes; 0.15)";

    public void Check(ShapeStep step)
    {
        ShapeGeometry.NoValue(step);
        _ = step.Number("shoulder", 0.5f, 0.1f, 0.9f);
        _ = step.Number("pad", 0.15f, 0, 20);
    }

    public ShapePlan Plan(ShapeStep step, ShapeContext context)
    {
        float shoulder = context.Y + step.Number("shoulder", 0.5f, 0.1f, 0.9f) * context.Height;
        float left = context.X, right = context.Right, middle = context.X + context.Width / 2, bottom = context.Bottom;
        float below = bottom - shoulder;

        // Each lower side a cubic from the shoulder to the point: straight down at first, flat into the point; every
        // control point inside the box, so the curve is too.
        int segments = Math.Clamp((int)Math.Ceiling(Math.Sqrt(Math.Max(context.Width, below) / (8 * Math.Max(1e-3f, context.Flatness))) * 2), 4, 512);
        var points = new List<float>(8 + 4 * segments) { left, context.Y, right, context.Y, right, shoulder };
        void Side(float x0, float x1, float x2, bool down)
        {
            // From (x0, shoulder) to (middle, bottom) going down, or back up.
            for (int s = 1; s <= segments; s++)
            {
                double t = down ? (double)s / segments : 1 - (double)s / segments;
                double u = 1 - t;
                double x = u * u * u * x0 + 3 * u * u * t * x1 + 3 * u * t * t * x2 + t * t * t * middle;
                double y = u * u * u * shoulder + 3 * u * u * t * (shoulder + 0.6 * below) + 3 * u * t * t * (shoulder + 0.84 * below) + t * t * t * bottom;
                if (!down && s == segments)
                {
                    break;                                          // the shoulder itself comes next, as the left side's start
                }

                points.Add((float)x);
                points.Add((float)y);
            }
        }

        Side(right, right, right - 0.25f * context.Width, down: true);
        Side(left, left, left + 0.25f * context.Width, down: false);
        points.Add(left);
        points.Add(shoulder);

        float inset = ShapeGeometry.Half(step, context) + ShapeGeometry.Pad(step, context, 0.15f);
        var region = ShapeGeometry.Region(0, 0, left, context.Y, right, shoulder, inset, inset, inset, inset) with { Paragraphs = step.HoldsParagraphs() };
        return new ShapePlan([ShapeGeometry.Path(step, [.. points], closed: true)], null, [region]);
    }
}

/// <summary>
/// "scribble": a hand's signature as a pen would draw it, never text: a capital's tall loop, then letter-like loops,
/// humps, ascenders and descenders in one running stroke, slanted, ending in a flourish; at times a short stroke across
/// it. Every choice comes from the context's seed; it fills its box, written from its start side.
/// </summary>
internal sealed class ScribbleShape : IShape
{
    // The letters are drawn whole by a profile.
    // Points a letter's loop is drawn with.
    private const int StepsPerLetter = 40;

    public string Name => "scribble";

    public string Summary => "a hand's signature-like scribble filling its box, never text: letters (how many letter-like loops; 4 to 9 from the seed), slant (degrees, forward "
        + "positive; 5 to 25 from the seed), flourish (none, under: a stroke back under it, loop: a last wide loop, any: one of them from the seed; any)";

    public void Check(ShapeStep step)
    {
        ShapeGeometry.NoValue(step);
        _ = step.Integer("letters", 6, 1, 40);
        _ = step.Number("slant", 15, -45, 45);
        _ = step.Choice("flourish", "any", "none", "under", "loop", "any");
    }

    public ShapePlan Plan(ShapeStep step, ShapeContext context)
    {
        if (!(context.Width > 1) || !(context.Height > 1))
        {
            return ShapePlan.Empty;
        }

        // Every draw made whether used or not, so a given option does not move the others.
        var random = new Random(context.Seed);
        int letters = step.Has("letters") ? step.Integer("letters", 6, 1, 40) : 4 + random.Next(6);
        double slantDrawn = 5 + 20 * random.NextDouble();
        double slant = (step.Has("slant") ? step.Number("slant", 15, -45, 45) : slantDrawn) * Math.PI / 180;
        string flourish = step.Choice("flourish", "any", "none", "under", "loop", "any");
        int flourishDrawn = random.Next(3);
        if (flourish == "any")
        {
            flourish = flourishDrawn switch { 0 => "none", 1 => "under", _ => "loop" };
        }

        bool cross = random.NextDouble() < 0.35;
        double drift = (random.NextDouble() - 0.5) * 0.12;

        // The running stroke, in units of the letters' height (y up): each letter one turn of a loop that rises and falls
        // back while the pen moves on; a loop wider than the pen's step crosses itself, as a hand's does.
        var stroke = new List<double>((letters + 3) * StepsPerLetter * 2);
        double x = 0;
        void Turn(double height, double width, double advance)
        {
            for (int s = 1; s <= StepsPerLetter; s++)
            {
                double theta = 2 * Math.PI * s / StepsPerLetter;
                double px = x + advance * s / StepsPerLetter + width * PortableMath.Sin(theta);
                double py = height * (1 - PortableMath.Cos(theta)) / 2;
                stroke.Add(px);
                stroke.Add(py);
            }

            x += advance;
        }

        stroke.Add(0);
        stroke.Add(0);
        Turn(2.4 + 0.8 * random.NextDouble(), 0.5 + 0.35 * random.NextDouble(), 1.2);         // the capital
        for (int k = 0; k < letters; k++)
        {
            double kind = random.NextDouble();
            double height = kind < 0.2 ? 1.8 + 0.6 * random.NextDouble()                       // an ascender
                : kind < 0.32 ? -(1.2 + 0.5 * random.NextDouble())                              // a descender
                : 0.7 + 0.5 * random.NextDouble();                                               // a small letter
            Turn(height, 0.15 + 0.4 * random.NextDouble(), 0.8 + 0.45 * random.NextDouble());
        }

        double endX = x;
        switch (flourish)
        {
            case "loop":
                Turn(1.6 + 0.6 * random.NextDouble(), 0.6, 1.2);
                break;
            case "under":
            {
                // Down past the baseline, then back under the letters: a quadratic from the last point.
                double fromX = stroke[^2], fromY = stroke[^1], toX = 0.4 * endX * random.NextDouble(), toY = -0.9 - 0.4 * random.NextDouble();
                double cx = endX + 0.8, cy = -1.6;
                for (int s = 1; s <= 2 * StepsPerLetter; s++)
                {
                    double t = (double)s / (2 * StepsPerLetter), u = 1 - t;
                    stroke.Add(u * u * fromX + 2 * u * t * cx + t * t * toX);
                    stroke.Add(u * u * fromY + 2 * u * t * cy + t * t * toY);
                }

                break;
            }
        }

        var paths = new List<double[]> { stroke.ToArray() };
        if (cross && letters > 2)
        {
            // A short stroke across the letters, as a crossed letter's.
            double at = endX * (0.3 + 0.4 * random.NextDouble()), length = 0.6 + 0.8 * random.NextDouble(), lift = 1.1 + 0.5 * random.NextDouble();
            paths.Add([at - length / 2, lift - 0.1, at + length / 2, lift + 0.15]);
        }

        // Slant and drift, then the whole fitted into the box (its aspect kept within a factor of 2), from the start side.
        double tan = PortableMath.Tan(slant);
        double minX = double.MaxValue, maxX = double.MinValue, minY = double.MaxValue, maxY = double.MinValue;
        foreach (var p in paths)
        {
            for (int i = 0; i < p.Length; i += 2)
            {
                p[i] += p[i + 1] * tan;
                p[i + 1] += drift * p[i];
                (minX, maxX, minY, maxY) = (Math.Min(minX, p[i]), Math.Max(maxX, p[i]), Math.Min(minY, p[i + 1]), Math.Max(maxY, p[i + 1]));
            }
        }

        double spanX = Math.Max(1e-6, maxX - minX), spanY = Math.Max(1e-6, maxY - minY);
        double sx = context.Width / spanX, sy = context.Height / spanY;
        (sx, sy) = (Math.Min(sx, 2 * sy), Math.Min(sy, 2 * sx));
        double offX = (context.Width - spanX * sx) / 2, offY = (context.Height - spanY * sy) / 2;
        var planned = new List<ShapePath>(paths.Count);
        foreach (var p in paths)
        {
            var points = new float[p.Length];
            for (int i = 0; i < p.Length; i += 2)
            {
                float along = (float)(offX + (p[i] - minX) * sx);
                points[i] = Math.Clamp(context.RightToLeft ? context.Right - along : context.X + along, context.X, context.Right);
                points[i + 1] = Math.Clamp((float)(context.Y + offY + (maxY - p[i + 1]) * sy), context.Y, context.Bottom);
            }

            planned.Add(ShapeGeometry.Path(step, points, closed: false));
        }

        return new ShapePlan(planned);
    }
}
