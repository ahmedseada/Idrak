// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak.Abstraction.Data;
using Idrak.Imaging;

// Plan 18, phase 3: text set in lines (wrapping, direction, alignment, fallback fonts) and pages drawn from parts by the
// built-in layouts ("letter", "plain"), each page's choices made from its seed.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] PageGroup =
    [
        ("pages: a paragraph wraps at spaces within the width, every line kept and none wider; justified lines fill it, start-aligned Arabic lines sit at the right", TextLayoutLines),
        ("pages: a right-to-left line with Latin in it draws the Latin first from the left, in its own order, and the glyphs advance left to right", TextLayoutVisualOrder),
        ("pages: characters the style's font lacks are set in the first fallback that has them", TextLayoutFallback),
    ];

    // A company's letter to a client about a delivery schedule: Arabic with Latin digits, a date, a long wrapping
    // paragraph, a lam-alef (لا), hamzas (أ, ء) and taa marbutas (ة).
    internal static readonly PageContent ArabicLetter = new()
    {
        HeaderStart = ["شركة النور للتوريدات", "إدارة خدمة العملاء"],
        HeaderEnd = ["الرقم: 12/345", "التاريخ: 2026/03/12"],
        Subject = "الموضوع: موعد تسليم الطلبية",
        Body =
        [
            "عزيزنا العميل، تحية طيبة وبعد: يسرنا إبلاغكم بأن طلبيتكم رقم 4521 المسجلة لدينا في عام 2026 قد أصبحت جاهزة للشحن، وسيتم تسليمها خلال ثلاثة أيام عمل من تاريخ هذا الخطاب إلى العنوان المسجل لدينا.",
            "نشكر لكم ثقتكم بنا، ونسعد بخدمتكم دائما.",
        ],
        Signature = ["مدير خدمة العملاء", "سارة بن أحمد"],
        Footer = ["العنوان: الرياض، هاتف 0112345678"],
        Language = "ar",
    };

    // A newsletter: a title and a few short paragraphs, no header or signature.
    internal static readonly PageContent Newsletter = new()
    {
        Title = "Community Garden News",
        Body =
        [
            "Spring planting starts on 4 April: bring gloves and a friend.",
            "The new compost bins are next to the north gate.",
            "Volunteers meet every Saturday at 9, rain or shine.",
            "Thank you to the 27 families who joined this year.",
        ],
        Footer = ["Issue 12, March 2026"],
        Language = "en",
    };

    internal static void TextLayoutLines(Device device)
    {
        var face = SubsetFace();
        string paragraph = "The quick brown fox jumps over the lazy dog while the team reviews the plan and the notes in turn, line by line.";
        var style = new TextStyle(face, 24);
        float width = 300;
        var block = TextLayout.Layout(paragraph, style, width);
        Check(block.Lines.Count > 2, $"{block.Lines.Count} lines in {width}px");
        Check(block.Lines.All(l => l.Width <= width + 0.01f), $"widths {string.Join(", ", block.Lines.Select(l => l.Width.ToString("F1")))} within {width}");
        Check(string.Join(" ", block.Lines.Select(l => l.Text)) == paragraph, "the lines keep every word, broken at spaces");
        Check(block.Lines.Zip(block.Lines.Skip(1)).All(p => p.Second.Baseline > p.First.Baseline), "each line under the one before");

        var justified = TextLayout.Layout(paragraph, style, width, TextAlignment.Justify);
        Check(justified.Lines.SkipLast(1).All(l => MathF.Abs(l.Width - width) < 0.5f), $"justified widths {string.Join(", ", justified.Lines.Select(l => l.Width.ToString("F1")))}");
        Check(justified.Lines[^1].Width < width - 1, "the last line is not stretched");

        string arabic = string.Join(" ", ArabicLetter.Body);
        var right = TextLayout.Layout(arabic, style, 400);
        Check(right.Lines.All(l => l.RightToLeft && MathF.Abs(l.Left + l.Width - 400) < 0.5f), "Arabic lines aligned at their start sit at the right");
        var centred = TextLayout.Layout(arabic, style, 400, TextAlignment.Center);
        Check(centred.Lines.All(l => MathF.Abs(2 * l.Left + l.Width - 400) < 0.5f), "centred lines sit in the middle");
        Check(TextLayout.Layout("", style, 400).Lines.Count == 0, "no text, no lines");
    }

    internal static void TextLayoutVisualOrder(Device device)
    {
        var face = SubsetFace();
        var line = TextLayout.Layout("عربي ABC نص", new TextStyle(face, 24)).Lines.Single();
        Check(line.RightToLeft, "the paragraph is right to left by its first strong character");
        var glyphs = line.Glyphs.Select(g => g.Glyph).ToArray();
        int a = face.Glyph('A'), b = face.Glyph('B'), c = face.Glyph('C');
        int at = Array.IndexOf(glyphs, a);
        Check(at >= 0 && glyphs[at + 1] == b && glyphs[at + 2] == c, "ABC keeps its own order");
        Check(Array.IndexOf(glyphs, face.Glyph('ن')) < 0 && glyphs.Length > 3, "Arabic letters are shaped into joined forms");
        Check(line.Glyphs.Zip(line.Glyphs.Skip(1)).All(p => p.Second.X >= p.First.X - 0.01f), "glyphs placed left to right");
        float shift = line.Glyphs[at].X;
        var style = new TextStyle(face, 24);
        float space = face.Advance(face.Glyph(' ')) * style.Scale(face);
        Check(MathF.Abs(shift - (TextLayout.Measure("نص", style) + space)) < 0.5f, "the last Arabic word (نص) comes first from the left, then a space, then ABC");
    }

    internal static void TextLayoutFallback(Device device)
    {
        var small = new OpenTypeFormat().Read(File.ReadAllBytes(FontData("features.ttf")), "features").Single();
        var full = SubsetFace();
        Check(small.Glyph('Ω') == 0 && full.Glyph('Ω') != 0, "the fixture: features.ttf has no Greek");
        var line = TextLayout.Layout("abc Ωμέγα", new TextStyle(small, 24) { Fallbacks = [full] }).Lines.Single();
        Check(line.Glyphs.Take(3).All(g => ReferenceEquals(g.Font, small)), "Latin in the style's font");
        Check(line.Glyphs.TakeLast(5).All(g => ReferenceEquals(g.Font, full) && g.Glyph != 0), "Greek in the fallback");
        var none = TextLayout.Layout("Ω", new TextStyle(small, 24)).Lines.Single();
        Check(none.Glyphs.Single().Glyph == 0, "no fallback: the font's missing glyph");
    }

    internal static PageStyle TestPageStyle(bool color = true) => new() { Fonts = [SubsetFace()], Color = color };

    internal static byte[][] Planes(ImageData image)
    {
        Check(image.TryGetPlanes(out var planes, out _), "a page is 8-bit planes");
        return [.. planes.Select(p => p.ToArray())];
    }

    // The page's lightness per pixel (the planes' mean), 0 black to 1 white.
    internal static float[] Grey(ImageData image)
    {
        var planes = Planes(image);
        var grey = new float[image.Width * image.Height];
        foreach (var plane in planes)
        {
            for (int i = 0; i < grey.Length; i++)
            {
                grey[i] += plane[i] / (255f * planes.Length);
            }
        }

        return grey;
    }

    internal static float Overlap(PageBlock a, PageBlock b) =>
        Math.Max(0, Math.Min(a.X + a.Width, b.X + b.Width) - Math.Max(a.X, b.X)) * Math.Max(0, Math.Min(a.Y + a.Height, b.Y + b.Height) - Math.Max(a.Y, b.Y));
}
