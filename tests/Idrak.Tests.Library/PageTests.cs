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
        ("pages: letter and plain draw every part (header, subject, body, signature, footer) inside the page without overlap, inked, in reading order, the start side following the text's direction", PagesDrawParts),
        ("pages: the same seed draws the same page, other seeds other choices; a grey style draws one plane; a long body makes a taller page with the footer still on it", PagesBySeed),
        ("pages: the layouts by name; an unknown name and a style with no font are refused; an app's layout replaces the library's of its name only", PageRegistry),
    ];

    internal static readonly PageContent ArabicLetter = new()
    {
        HeaderStart = ["المملكة العربية السعودية", "وزارة العدل"],
        HeaderEnd = ["الرقم: 12/345", "التاريخ: 1447/03/12"],
        Subject = "الموضوع: طلب نسخة من الحكم",
        Body =
        [
            "سعادة رئيس المحكمة العامة المحترم، السلام عليكم ورحمة الله وبركاته، وبعد: نفيدكم بأن المدعي تقدم بطلب نسخة من الحكم الصادر في القضية رقم 4521 لعام 1446 هـ.",
            "نأمل التكرم بالاطلاع والإفادة، شاكرين لكم حسن تعاونكم.",
        ],
        Signature = ["مدير الإدارة", "أحمد بن سعد"],
        Footer = ["العنوان: الرياض، هاتف 0112345678"],
        Language = "ar",
    };

    internal static readonly PageContent EnglishLetter = new()
    {
        HeaderStart = ["Ministry of Justice", "General Court"],
        HeaderEnd = ["No. 12/345", "Date: 2026-03-12"],
        Subject = "Subject: a copy of the judgment",
        Body = ["Dear Sir, we would be grateful to receive a copy of the judgment issued in case 4521 of 2025, for the claimant's records.", "Kind regards."],
        Signature = ["Head of Department", "A. Saad"],
        Footer = ["Riyadh, phone 0112345678"],
        Language = "en",
    };

    internal static void TextLayoutLines(Device device)
    {
        var face = SubsetFace();
        string paragraph = "The quick brown fox jumps over the lazy dog while the court reads the claim and the answer in turn, line by line.";
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
        var small = FontFormats.Read(FontData("features.ttf")).Single();
        var full = SubsetFace();
        Check(small.Glyph('Ω') == 0 && full.Glyph('Ω') != 0, "the fixture: features.ttf has no Greek");
        var line = TextLayout.Layout("abc Ωμέγα", new TextStyle(small, 24) { Fallbacks = [full] }).Lines.Single();
        Check(line.Glyphs.Take(3).All(g => ReferenceEquals(g.Font, small)), "Latin in the style's font");
        Check(line.Glyphs.TakeLast(5).All(g => ReferenceEquals(g.Font, full) && g.Glyph != 0), "Greek in the fallback");
        var none = TextLayout.Layout("Ω", new TextStyle(small, 24)).Lines.Single();
        Check(none.Glyphs.Single().Glyph == 0, "no fallback: the font's missing glyph");
    }

    internal static PageStyle TestPageStyle(bool color = true) => new() { Fonts = [SubsetFace()], Color = color };

    internal static void PagesDrawParts(Device device)
    {
        var style = TestPageStyle();
        string[] textKinds = ["header", "title", "subject", "body", "signature", "footer"];
        var watch = Stopwatch.StartNew();
        int pages = 0;
        foreach (string name in new[] { "letter", "plain" })
        {
            foreach (var (content, rtl) in new[] { (ArabicLetter, true), (EnglishLetter, false) })
            {
                for (int seed = 1; seed <= 4; seed++)
                {
                    var page = PageLayouts.Get(name).Render(content, style, seed);
                    pages++;
                    string where = $"{name}, {(rtl ? "Arabic" : "English")}, seed {seed} ({page.Description})";
                    var image = page.Image;
                    Check(image.Width == style.Width && image.Height >= style.Height && image.Channels == 3, $"{where}: {image.Channels}x{image.Height}x{image.Width}");
                    Check(page.Text == content.ReadingOrder(), $"{where}: the text is the content's reading order");

                    var text = page.Blocks.Where(b => textKinds.Contains(b.Kind)).ToList();
                    var expected = (name == "plain" ? new[] { "header" } : ["header", "header"]).Concat(["subject"])
                        .Concat(name == "plain" ? ["body"] : ["body", "body"]).Concat(["signature", "footer"]);
                    Check(text.Select(b => b.Kind).SequenceEqual(expected), $"{where}: parts {string.Join(",", text.Select(b => b.Kind))}");
                    Check(string.Join("\n", text.Select(b => b.Text)) == page.Text, $"{where}: the parts' text in reading order");

                    var grey = Grey(image);
                    foreach (var block in text)
                    {
                        Check(block.X >= 0 && block.Y >= 0 && block.X + block.Width <= image.Width + 0.5f && block.Y + block.Height <= image.Height + 0.5f,
                            $"{where}: {block.Kind} at ({block.X:F0},{block.Y:F0}) {block.Width:F0}x{block.Height:F0} inside the page");
                        Check(Darkest(grey, image.Width, block) < 0.5f, $"{where}: {block.Kind} is inked");
                    }

                    for (int i = 0; i < text.Count; i++)
                    {
                        for (int j = i + 1; j < text.Count; j++)
                        {
                            Check(Overlap(text[i], text[j]) < 1, $"{where}: {text[i].Kind} and {text[j].Kind} overlap");
                        }
                    }

                    var start = text[0];
                    Check(name == "plain" || rtl == start.X + start.Width / 2 > image.Width / 2f, $"{where}: the first header part is on the {(rtl ? "right" : "left")}");
                }
            }
        }

        Console.WriteLine($"    {pages} pages, {watch.Elapsed.TotalMilliseconds / pages:F0} ms a page");
    }

    internal static void PagesBySeed(Device device)
    {
        var style = TestPageStyle();
        var letter = PageLayouts.Get("letter");
        var first = letter.Render(ArabicLetter, style, 7);
        var again = letter.Render(ArabicLetter, style, 7);
        Check(first.Description == again.Description && Planes(first.Image).Zip(Planes(again.Image)).All(p => p.First.SequenceEqual(p.Second)), "seed 7 twice: the same page");
        var descriptions = Enumerable.Range(1, 8).Select(s => letter.Render(ArabicLetter, style, s).Description).Distinct().Count();
        Check(descriptions >= 4, $"8 seeds, {descriptions} different choices");

        var greyPage = letter.Render(ArabicLetter, TestPageStyle(color: false), 7);
        Check(greyPage.Image.Channels == 1, "a grey style: one plane");

        var none = new PageStyle { Fonts = [SubsetFace()], Rule = 0, Emblem = 0, Stamp = 0 };
        var plainLetter = letter.Render(ArabicLetter, none, 3);
        Check(!plainLetter.Blocks.Any(b => b.Kind is "rule" or "emblem" or "stamp"), $"no rule, emblem or stamp when their chances are 0 ({plainLetter.Description})");
        var all = new PageStyle { Fonts = [SubsetFace()], Rule = 1, Emblem = 1, Stamp = 1 };
        var ornate = letter.Render(ArabicLetter, all, 3);
        Check(new[] { "rule", "emblem", "stamp" }.All(k => ornate.Blocks.Any(b => b.Kind == k)), $"each when its chance is 1 ({ornate.Description})");

        var longBody = ArabicLetter with { Body = [.. Enumerable.Repeat(ArabicLetter.Body[0], 30)] };
        foreach (string name in new[] { "letter", "plain" })
        {
            var tall = PageLayouts.Get(name).Render(longBody, style, 2);
            var footer = tall.Blocks.Single(b => b.Kind == "footer");
            var lastBody = tall.Blocks.Last(b => b.Kind == "body");
            Check(tall.Image.Height > style.Height && footer.Y + footer.Height <= tall.Image.Height && footer.Y > lastBody.Y + lastBody.Height,
                $"{name}: a long body, a page {tall.Image.Height}px tall with the footer under the body");
        }
    }

    internal static void PageRegistry(Device device)
    {
        Check(PageLayouts.Names.Contains("letter") && PageLayouts.Names.Contains("plain"), $"names {string.Join(", ", PageLayouts.Names)}");
        Throws<NotSupportedException>(() => PageLayouts.Get("nope"), "an unknown layout");
        Throws<ArgumentException>(() => PageLayouts.Get("letter").Render(ArabicLetter, new PageStyle { Fonts = [] }, 1), "a style with no font");

        PageLayouts.Register(new BlankLayout());
        try
        {
            Check(PageLayouts.Get("plain").Render(ArabicLetter, TestPageStyle(), 1).Description == "blank", "the app's plain draws");
            Check(PageLayouts.Get("letter").Name == "letter" && PageLayouts.Default("plain") is { } library && library.Name == "plain", "the letter, and the library's plain behind the app's, stay");
        }
        finally
        {
            PageLayouts.Unregister("plain");
        }

        Check(PageLayouts.Get("plain").Render(ArabicLetter, TestPageStyle(), 1).Description.StartsWith("plain:", StringComparison.Ordinal), "unregistered: the library's plain again");
    }

    internal sealed class BlankLayout : IPageLayout
    {
        public string Name => "plain";

        public string Summary => "blank";

        public RenderedPage Render(PageContent content, PageStyle style, int seed) =>
            new(new ImageData(new float[style.Width * style.Height], 1, style.Height, style.Width), content.ReadingOrder(), [], "blank");
    }

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

    internal static float Darkest(float[] grey, int width, PageBlock block)
    {
        float least = 1;
        int height = grey.Length / width;
        for (int y = Math.Max(0, (int)block.Y); y < Math.Min(height, (int)MathF.Ceiling(block.Y + block.Height)); y++)
        {
            for (int x = Math.Max(0, (int)block.X); x < Math.Min(width, (int)MathF.Ceiling(block.X + block.Width)); x++)
            {
                least = Math.Min(least, grey[y * width + x]);
            }
        }

        return least;
    }

    internal static float Overlap(PageBlock a, PageBlock b) =>
        Math.Max(0, Math.Min(a.X + a.Width, b.X + b.Width) - Math.Max(a.X, b.X)) * Math.Max(0, Math.Min(a.Y + a.Height, b.Y + b.Height) - Math.Max(a.Y, b.Y));
}
