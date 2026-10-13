// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Globalization;
using Idrak.Imaging.Shaping;
using static Idrak.Imaging.PageDrawing;

namespace Idrak.Imaging;

/// <summary>
/// Where a page is drawn: an ink layer (text and rules) and a blue layer (signatures and stamps) of coverage, composed
/// on tinted paper into 8-bit planes. The layers are pooled and given back when the page is made.
/// </summary>
internal sealed class PageCanvas : IDisposable
{
    private float[]? _ink, _blue;

    public PageCanvas(int width, int height)
    {
        Width = width;
        Height = height;
        _ink = ArrayPool<float>.Shared.Rent(width * height);
        Ink.Clear();
    }

    public int Width { get; }

    public int Height { get; }

    public Span<float> Ink => _ink.AsSpan(0, Width * Height);

    public Span<float> Blue
    {
        get
        {
            if (_blue is null)
            {
                _blue = ArrayPool<float>.Shared.Rent(Width * Height);
                _blue.AsSpan(0, Width * Height).Clear();
            }

            return _blue.AsSpan(0, Width * Height);
        }
    }

    /// <summary>A filled rectangle of ink (a rule).</summary>
    public void Rectangle(Span<float> layer, float x, float y, float w, float h)
    {
        var rect = new GlyphOutline([OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close], [0, 0, w, 0, w, h, 0, h]);
        OutlineRasterizers.Get().Fill(rect, 1f, x, y + h, layer, Width, Height);
    }

    /// <summary>A ring (a stamp's or emblem's circle) of the given radius and stroke, as two polygons wound opposite ways.</summary>
    public void Ring(Span<float> layer, float cx, float cy, float radius, float stroke)
    {
        const int Steps = 96;
        var verbs = new List<OutlineVerb>();
        var points = new List<float>();
        foreach (var (r, dir) in ((float, int)[])[(radius, 1), (radius - stroke, -1)])
        {
            for (int k = 0; k <= Steps; k++)
            {
                double a = dir * 2 * Math.PI * k / Steps;
                verbs.Add(k == 0 ? OutlineVerb.Move : OutlineVerb.Line);
                points.Add((float)(r * Math.Cos(a)));
                points.Add((float)(r * Math.Sin(a)));
            }

            verbs.Add(OutlineVerb.Close);
        }

        OutlineRasterizers.Get().Fill(new GlyphOutline([.. verbs], [.. points]), 1f, cx, cy, layer, Width, Height);
    }

    /// <summary>The page: paper of the given colour, the ink layer laid on at its darkness, the blue layer in its colour.</summary>
    public ImageData Compose((float R, float G, float B) paper, float darkness, (float R, float G, float B) blue, bool color)
    {
        int size = Width * Height;
        int channels = color ? 3 : 1;
        var planes = new byte[channels][];
        for (int c = 0; c < channels; c++)
        {
            planes[c] = new byte[size];
        }

        var ink = Ink;
        var blueLayer = _blue is null ? default : _blue.AsSpan(0, size);
        float[] paperOf = color ? [paper.R, paper.G, paper.B] : [(paper.R + paper.G + paper.B) / 3];
        float[] blueOf = color ? [blue.R, blue.G, blue.B] : [(blue.R + blue.G + blue.B) / 3];
        for (int c = 0; c < channels; c++)
        {
            var plane = planes[c];
            float p = paperOf[c], inkTone = p * (1 - darkness), b = blueOf[c];
            for (int i = 0; i < size; i++)
            {
                float v = p + (inkTone - p) * ink[i];
                if (!blueLayer.IsEmpty)
                {
                    v += (b - v) * blueLayer[i];
                }

                plane[i] = (byte)Math.Clamp((int)MathF.Round(v * 255f), 0, 255);
            }
        }

        var levels = new float[256];
        for (int k = 0; k < 256; k++)
        {
            levels[k] = k / 255f;
        }

        return new ImageData(planes, Height, Width, levels);
    }

    public void Dispose()
    {
        if (_ink is not null)
        {
            ArrayPool<float>.Shared.Return(_ink);
            _ink = null;
        }

        if (_blue is not null)
        {
            ArrayPool<float>.Shared.Return(_blue);
            _blue = null;
        }
    }
}

/// <summary>What the built-in layouts share: the page's choices from its seed, text blocks, and the composition.</summary>
internal static class PageDrawing
{
    internal sealed class Choices(PageStyle style, int seed)
    {
        public readonly Random Random = new(seed);

        public float Between((float Min, float Max) range) => range.Min + (float)Random.NextDouble() * (range.Max - range.Min);

        public float Between(float min, float max) => min + (float)Random.NextDouble() * (max - min);

        public bool Chance(float p) => Random.NextDouble() < p;

        public IFontFace Font() => style.Fonts[Random.Next(style.Fonts.Count)];
    }

    internal static bool Direction(PageContent content) =>
        content.RightToLeft ?? Bidi.Resolve(content.ReadingOrder().Replace('\n', ' ')).RightToLeft;

    internal static TextStyle Style(PageStyle style, IFontFace font, float size, float spacing, string? language) =>
        new(font, size) { Fallbacks = style.Fallbacks, LineSpacing = spacing, Language = language };

    // A block of lines laid out in `width`, each line a paragraph.
    internal static TextBlock Block(IEnumerable<string> lines, TextStyle style, float width, TextAlignment alignment, bool rtl) =>
        TextLayout.Layout(string.Join("\n", lines), style, width, alignment, rtl);

    internal static string Describe(TextStyle style) =>
        string.Create(CultureInfo.InvariantCulture, $"{style.Font.Family} {style.Font.Style} {style.Size:F1}px");

    internal static (float R, float G, float B) Paper(Choices c, bool color) =>
        color ? (1f, c.Between(0.97f, 1f), c.Between(0.92f, 1f)) : (1f, 1f, 1f);

    internal static (float R, float G, float B) BlueInk(Choices c) => (c.Between(0.05f, 0.2f), c.Between(0.15f, 0.3f), c.Between(0.45f, 0.7f));
}

/// <summary>
/// "letter": an official letter. A header of three places (the start side's state and ministry lines, an emblem ring or
/// centre lines, the end side's number, date and attachments), a rule, an optional centred title, the subject line, the
/// body (justified or at its start), a signature block at the end side (blue ink or black), an optional stamp ring, and a
/// footer under a rule. Every size, margin and spacing is drawn from the style's ranges.
/// </summary>
internal sealed class LetterLayout : IPageLayout
{
    public string Name => "letter";

    public string Summary => "an official letter: three-place header, rule, title, subject, body, signature block, optional stamp, footer";

    public RenderedPage Render(PageContent content, PageStyle style, int seed)
    {
        ArgumentNullException.ThrowIfNull(content);
        ArgumentNullException.ThrowIfNull(style);
        if (style.Fonts.Count == 0)
        {
            throw new ArgumentException("A page needs at least one font.", nameof(style));
        }

        var c = new Choices(style, seed);
        bool rtl = Direction(content);
        int width = style.Width;
        float margin = width * c.Between(style.Margin);
        float inner = width - 2 * margin;
        var bodyFont = c.Font();
        var headerFont = c.Chance(style.OtherHeaderFont) ? c.Font() : bodyFont;
        float bodySize = c.Between(style.BodySize), spacing = c.Between(style.LineSpacing);
        var body = Style(style, bodyFont, bodySize, spacing, content.Language);
        var header = Style(style, headerFont, bodySize * c.Between(0.8f, 1.0f), c.Between(1.0f, 1.3f), content.Language);
        var title = Style(style, headerFont, bodySize * c.Between(1.1f, 1.4f), 1.1f, content.Language);
        var footer = Style(style, bodyFont, bodySize * c.Between(0.6f, 0.8f), 1.1f, content.Language);
        bool justify = c.Chance(style.Justify);

        // The parts, laid out first so the page's height follows from them.
        var placed = new List<(string Kind, string Text, TextBlock Block, float X, float Y, bool Blue)>();
        float y = margin * c.Between(0.5f, 0.9f);
        float side = inner * c.Between(0.32f, 0.4f), centre = inner - 2 * side;
        float startX = rtl ? width - margin - side : margin, endX = rtl ? margin : width - margin - side;
        float headerBottom = y;
        bool emblem = content.HeaderCenter.Count == 0 && c.Chance(style.Emblem);
        foreach (var (lines, x, w) in ((IReadOnlyList<string>, float, float)[])[(content.HeaderStart, startX, side), (content.HeaderCenter, margin + side, centre), (content.HeaderEnd, endX, side)])
        {
            if (lines.Count == 0)
            {
                continue;
            }

            var alignment = ReferenceEquals(lines, content.HeaderEnd) && c.Chance(0.5f) ? TextAlignment.Start : TextAlignment.Center;
            var block = Block(lines, header, w, alignment, rtl);
            placed.Add(("header", string.Join("\n", lines), block, x, y, false));
            headerBottom = Math.Max(headerBottom, y + block.Height);
        }

        float emblemRadius = Math.Min(centre * 0.3f, bodySize * 2.2f);
        if (emblem)
        {
            headerBottom = Math.Max(headerBottom, y + 2 * emblemRadius + 4);
        }

        y = headerBottom + bodySize * c.Between(0.4f, 1.0f);
        bool rule = c.Chance(style.Rule);
        float ruleY = y, ruleThickness = c.Between(1f, 3f);
        if (rule)
        {
            y += ruleThickness + bodySize * c.Between(0.5f, 1.0f);
        }

        void Add(string kind, IReadOnlyList<string> lines, TextStyle s, float x, float w, TextAlignment alignment, float gapAfter, bool blue = false)
        {
            if (lines.Count == 0)
            {
                return;
            }

            var block = Block(lines, s, w, alignment, rtl);
            placed.Add((kind, string.Join("\n", lines), block, x, y, blue));
            y += block.Height + gapAfter;
        }

        if (content.Title is { } t)
        {
            Add("title", [t], title, margin, inner, TextAlignment.Center, bodySize * c.Between(0.6f, 1.2f));
        }

        if (content.Subject is { } subject)
        {
            Add("subject", [subject], body, margin, inner, TextAlignment.Start, bodySize * c.Between(0.5f, 1.0f));
        }

        float paragraphGap = bodySize * c.Between(0.1f, 0.7f);
        foreach (string paragraph in content.Body)
        {
            Add("body", [paragraph], body, margin, inner, justify ? TextAlignment.Justify : TextAlignment.Start, paragraphGap);
        }

        y += bodySize * c.Between(0.8f, 2.0f);
        float signatureWidth = inner * c.Between(0.3f, 0.45f);
        float signatureX = rtl ? margin + inner * c.Between(0f, 0.1f) : width - margin - signatureWidth - inner * c.Between(0f, 0.1f);
        bool blueSignature = style.Color && c.Chance(style.BlueSignature);
        float signatureTop = y;
        Add("signature", content.Signature, body, signatureX, signatureWidth, TextAlignment.Center, bodySize, blueSignature);
        bool stamp = c.Chance(style.Stamp) && content.Signature.Count > 0;
        float stampRadius = bodySize * c.Between(2.0f, 3.2f);

        float footerHeight = 0;
        TextBlock? footerBlock = null;
        if (content.Footer.Count > 0)
        {
            footerBlock = Block(content.Footer, footer, inner, TextAlignment.Center, rtl);
            footerHeight = footerBlock.Height + bodySize;
        }

        int height = (int)Math.Ceiling(Math.Max(style.Height, y + footerHeight + margin * 0.6f));
        using var canvas = new PageCanvas(width, height);
        var blocks = new List<PageBlock>();
        foreach (var (kind, text, block, x, top, blue) in placed)
        {
            TextRenderer.Draw(block, x, top, blue ? canvas.Blue : canvas.Ink, width, height);
            blocks.Add(new PageBlock(kind, text, x, top, block.Width, block.Height));
        }

        if (emblem)
        {
            float cx = width / 2f, cy = margin * 0.7f + emblemRadius + 2;
            canvas.Ring(canvas.Ink, cx, cy, emblemRadius, Math.Max(1.5f, emblemRadius * 0.08f));
            canvas.Ring(canvas.Ink, cx, cy, emblemRadius * 0.7f, Math.Max(1f, emblemRadius * 0.05f));
            blocks.Add(new PageBlock("emblem", "", cx - emblemRadius, cy - emblemRadius, 2 * emblemRadius, 2 * emblemRadius));
        }

        if (rule)
        {
            canvas.Rectangle(canvas.Ink, margin, ruleY, inner, ruleThickness);
            blocks.Add(new PageBlock("rule", "", margin, ruleY, inner, ruleThickness));
        }

        if (stamp)
        {
            float cx = signatureX + signatureWidth * c.Between(0.2f, 0.8f), cy = signatureTop + c.Between(-0.5f, 1.0f) * stampRadius;
            var layer = style.Color ? canvas.Blue : canvas.Ink;
            canvas.Ring(layer, cx, cy, stampRadius, Math.Max(2f, stampRadius * 0.07f));
            canvas.Ring(layer, cx, cy, stampRadius * 0.72f, Math.Max(1.5f, stampRadius * 0.04f));
            blocks.Add(new PageBlock("stamp", "", cx - stampRadius, cy - stampRadius, 2 * stampRadius, 2 * stampRadius));
        }

        if (footerBlock is not null)
        {
            float footerTop = height - margin * 0.6f - footerBlock.Height;
            canvas.Rectangle(canvas.Ink, margin, footerTop - bodySize * 0.5f, inner, 1.5f);
            TextRenderer.Draw(footerBlock, margin, footerTop, canvas.Ink, width, height);
            blocks.Add(new PageBlock("footer", string.Join("\n", content.Footer), margin, footerTop, inner, footerBlock.Height));
        }

        var image = canvas.Compose(Paper(c, style.Color), c.Between(style.Ink), BlueInk(c), style.Color);
        string description = string.Create(CultureInfo.InvariantCulture,
            $"letter: body {Describe(body)}, header {Describe(header)}, spacing {spacing:F2}, margin {margin:F0}px, {(justify ? "justified" : "start")}{(emblem ? ", emblem" : "")}{(rule ? ", rule" : "")}{(stamp ? ", stamp" : "")}{(blueSignature ? ", blue signature" : "")}");
        return new RenderedPage(image, content.ReadingOrder(), blocks, description);
    }
}

/// <summary>"plain": header lines at the start, a centred title, the subject, the body, centred footer lines; no rule or ornament.</summary>
internal sealed class PlainLayout : IPageLayout
{
    public string Name => "plain";

    public string Summary => "a plain page: header lines, title, subject, body, footer; no rule or ornament";

    public RenderedPage Render(PageContent content, PageStyle style, int seed)
    {
        ArgumentNullException.ThrowIfNull(content);
        ArgumentNullException.ThrowIfNull(style);
        if (style.Fonts.Count == 0)
        {
            throw new ArgumentException("A page needs at least one font.", nameof(style));
        }

        var c = new Choices(style, seed);
        bool rtl = Direction(content);
        int width = style.Width;
        float margin = width * c.Between(style.Margin), inner = width - 2 * margin;
        var font = c.Font();
        float size = c.Between(style.BodySize), spacing = c.Between(style.LineSpacing);
        var body = Style(style, font, size, spacing, content.Language);
        bool justify = c.Chance(style.Justify);
        var parts = new List<(string Kind, IReadOnlyList<string> Lines, TextStyle Style, TextAlignment Alignment)>
        {
            ("header", [.. content.HeaderStart, .. content.HeaderCenter, .. content.HeaderEnd], body with { Size = size * 0.9f }, TextAlignment.Start),
            ("title", content.Title is { } titleText ? [titleText] : [], body with { Size = size * 1.25f }, TextAlignment.Center),
            ("subject", content.Subject is { } subjectText ? [subjectText] : [], body, TextAlignment.Start),
            ("body", content.Body, body, justify ? TextAlignment.Justify : TextAlignment.Start),
            ("signature", content.Signature, body, TextAlignment.End),
            ("footer", content.Footer, body with { Size = size * 0.75f }, TextAlignment.Center),
        };

        var placed = new List<(string Kind, string Text, TextBlock Block, float Y)>();
        float y = margin * c.Between(0.5f, 1.0f);
        foreach (var (kind, lines, s, alignment) in parts)
        {
            if (lines.Count == 0)
            {
                continue;
            }

            var block = Block(lines, s, inner, alignment, rtl);
            placed.Add((kind, string.Join("\n", lines), block, y));
            y += block.Height + size * c.Between(0.4f, 1.0f);
        }

        int height = (int)Math.Ceiling(Math.Max(style.Height, y + margin * 0.6f));
        using var canvas = new PageCanvas(width, height);
        var blocks = new List<PageBlock>();
        foreach (var (kind, text, block, top) in placed)
        {
            TextRenderer.Draw(block, margin, top, canvas.Ink, width, height);
            blocks.Add(new PageBlock(kind, text, margin, top, inner, block.Height));
        }

        var image = canvas.Compose(Paper(c, style.Color), c.Between(style.Ink), BlueInk(c), style.Color);
        return new RenderedPage(image, content.ReadingOrder(), blocks,
            string.Create(CultureInfo.InvariantCulture, $"plain: {Describe(body)}, spacing {spacing:F2}, margin {margin:F0}px, {(justify ? "justified" : "start")}"));
    }
}
