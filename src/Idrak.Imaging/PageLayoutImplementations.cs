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

    // The default rasterizer, reached directly.
    private static readonly CoverageRasterizer CoverageRasterizer = new();

    /// <summary>A filled rectangle of ink (a rule).</summary>
    public void Rectangle(Span<float> layer, float x, float y, float w, float h)
    {
        var rect = new GlyphOutline([OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close], [0, 0, w, 0, w, h, 0, h]);
        CoverageRasterizer.Fill(rect, 1f, x, y + h, layer, Width, Height);
    }

    /// <summary>
    /// An outline (pixels, y up as outlines are: points (x, -y)) laid on <paramref name="layer"/> at <paramref name="share"/>
    /// of full ink (a pen's pressure, a tint's tone): ink over ink, as the rasterizer composites.
    /// </summary>
    public void Fill(Span<float> layer, GlyphOutline outline, float share)
    {
        if (outline.IsEmpty || !(share > 0))
        {
            return;
        }

        IOutlineRasterizer fill = CoverageRasterizer;
        if (share >= 1f)
        {
            fill.Fill(outline, 1f, 0, 0, layer, Width, Height);
            return;
        }

        // Its coverage in a box of its own first (pooled), then laid on at the share, row by row.
        var (minX, minY, maxX, maxY) = outline.Bounds();
        int left = Math.Max(0, (int)MathF.Floor(minX) - 1), right = Math.Min(Width, (int)MathF.Ceiling(maxX) + 2);
        int top = Math.Max(0, (int)MathF.Floor(-maxY) - 1), bottom = Math.Min(Height, (int)MathF.Ceiling(-minY) + 2);
        if (right <= left || bottom <= top)
        {
            return;
        }

        int w = right - left, h = bottom - top;
        float[] own = ArrayPool<float>.Shared.Rent(w * h);
        try
        {
            var coverage = own.AsSpan(0, w * h);
            coverage.Clear();
            fill.Fill(outline, 1f, -left, -top, coverage, w, h);
            for (int row = 0; row < h; row++)
            {
                var source = coverage.Slice(row * w, w);
                var target = layer.Slice((top + row) * Width + left, w);
                for (int x = 0; x < w; x++)
                {
                    float c = source[x] * share, t = target[x];
                    target[x] = t + c - t * c;
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(own);
        }
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

    internal static TextStyle Style(PageStyle style, IFontFace font, float size, float spacing, string? language, IReadOnlyList<string>? hand = null, int seed = 0) =>
        new(font, size) { Fallbacks = style.Fallbacks, LineSpacing = spacing, Language = language, Features = Asked(font, hand), Seed = Mix(seed, AlternateSeed) };

    // The hand's features a page asks for that the font's own GSUB lists, each turned on ("+PAIR"), the hand's alternates
    // picked per glyph ("ALTS=random"); null for none. Read from the font, not from the map that made it, so a mapped face
    // and the font converted from it (FontCache), or a made font and the file written from it, ask the same.
    internal static string[]? Asked(IFontFace font, IReadOnlyList<string>? hand)
    {
        if (hand is null || hand.Count == 0)
        {
            return null;
        }

        var substitutions = FontLayout.For(font).Substitutions;
        var on = hand.Where(substitutions.HasFeature).Select(tag => tag == FontMap.AlternatesFeature ? tag + "=random" : "+" + tag).ToArray();
        return on.Length == 0 ? null : on;
    }

    // The part numbers of a page's second-style and whole-word choices (Mix), as PairSeed is the pairs'; of its alternates'
    // chance and of the seed its glyphs' alternates are picked from.
    internal const int SecondStyleSeed = 7, WordSeed = 8, AlternatesSeed = 9, AlternateSeed = 10;

    // The hand's features of the page of `seed`: letter pairs, their second style and whole words, each at its own chance
    // (FontMapDrawing.Pairs, SecondStyle, Words) from its own part of the seed, the same for the same seed.
    internal static string[] Hand(int seed)
    {
        var drawing = FontMaps.Drawing;
        var on = new List<string>(3);
        if (Pairs(seed))
        {
            on.Add(FontMap.PairFeature);
        }

        if (drawing.SecondStyle > 0 && Mix(seed, SecondStyleSeed) / (double)int.MaxValue < drawing.SecondStyle)
        {
            on.Add(FontMap.SecondStyleFeature);
        }

        if (drawing.Words > 0 && Mix(seed, WordSeed) / (double)int.MaxValue < drawing.Words)
        {
            on.Add(FontMap.WordFeature);
        }

        if (drawing.Alternates >= 1 || drawing.Alternates > 0 && Mix(seed, AlternatesSeed) / (double)int.MaxValue < drawing.Alternates)
        {
            on.Add(FontMap.AlternatesFeature);
        }

        return [.. on];
    }

    // The part number of a page's letter-pair choice (Mix): hashed from the layout's seed, never drawn from its Random.
    internal const int PairSeed = 6;

    // Whether the page of `seed` joins letter pairs: at the font maps' chance (FontMapDrawing.Pairs), the same for the same seed.
    internal static bool Pairs(int seed)
    {
        double chance = FontMaps.Drawing.Pairs;
        return chance > 0 && Mix(seed, PairSeed) / (double)int.MaxValue < chance;
    }

    // A page whose text joins letter pairs, writes their second style or whole words says so in its description; one that
    // does none says nothing more.
    internal static string Paired(TextStyle style) => style.Features is null ? "" : string.Concat(style.Features.Select(f => f[1..] switch
    {
        FontMap.PairFeature => ", pairs",
        FontMap.SecondStyleFeature => ", second style",
        FontMap.WordFeature => ", words",
        _ when f.StartsWith(FontMap.AlternatesFeature, StringComparison.Ordinal) => ", alternates",
        _ => "",
    }));

    // A block of lines laid out in `width`, each line a paragraph.
    internal static TextBlock Block(IEnumerable<string> lines, TextStyle style, float width, TextAlignment alignment, bool rtl) =>
        TextLayout.Layout(string.Join("\n", lines), style, width, alignment, rtl);

    internal static string Describe(TextStyle style) =>
        string.Create(CultureInfo.InvariantCulture, $"{style.Font.Family} {style.Font.Style} {style.Size:F1}px");

    internal static (float R, float G, float B) Paper(Choices c, bool color) =>
        color ? (1f, c.Between(0.97f, 1f), c.Between(0.92f, 1f)) : (1f, 1f, 1f);

    internal static (float R, float G, float B) BlueInk(Choices c) => (c.Between(0.05f, 0.2f), c.Between(0.15f, 0.3f), c.Between(0.45f, 0.7f));

    // The part number of a page's glyph poses (Mix): their seed is hashed from the layout's, never drawn from its Random.
    internal const int GlyphSeed = 5;

    // The part numbers of a letter's emblem, stamp and signature choices (Mix): hashed from the layout's seed, never drawn
    // from its Random, so a page without them is drawn as it was.
    internal const int EmblemSeed = 11, StampSeed = 12, SignatureSeed = 13;

    // A posed page's description names its pose; a page as set says nothing more.
    internal static string Posed(GlyphPainter glyphs) => glyphs.IsIdentity ? "" : $", pose {glyphs.Name}";

    // A seed of its own for each part of a page's choices, from the page's seed and the part's number (SplitMix64; the
    // same in every process), so one part's draws never move another's.
    internal static int Mix(int seed, int part)
    {
        ulong z = ((ulong)(uint)seed << 32 | (uint)part) + 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return (int)((z ^ (z >> 31)) & int.MaxValue);
    }
}

/// <summary>
/// "letter": an official letter. A header of three places (the start side's organisation lines, an emblem or centre
/// lines, the end side's number, date and attachments), a rule, an optional centred title, the subject line, the body
/// (justified or at its start), a signature block at the end side with a pen-drawn signature over its name (blue ink or
/// black), an optional stamp over it, and a footer under a rule. Every size, margin and spacing is drawn from the
/// style's ranges. The emblem, the stamp and the signature are shape profiles (<c>ShapeProfiles</c>: one of the
/// page's <c>emblem-shapes</c>, <c>stamp-shapes</c> and <c>signature-shapes</c> options, the library's by default)
/// drawn through the shape pipeline by a pen (<c>emblem-pen</c>, <c>stamp-pen</c>, <c>signature-pen</c>); their texts
/// (the page's first header line, a stamp's generic word) are ornament, not the page's text. Their choices come from
/// seeds of their own, so a page without them is drawn as it was.
/// </summary>
public sealed class LetterLayout : IPageLayout
{
    /// <summary>"letter".</summary>
    public string Name => "letter";

    /// <summary>One line on what it draws.</summary>
    public string Summary => "an official letter: three-place header with an optional emblem, rule, title, subject, body, signature block with a pen-drawn signature, optional stamp, footer";

    /// <inheritdoc/>
    public RenderedPage Render(PageContent content, PageStyle style, int seed)
    {
        ArgumentNullException.ThrowIfNull(content);
        ArgumentNullException.ThrowIfNull(style);
        if (style.Fonts.Count == 0)
        {
            throw new ArgumentException("A page needs at least one font.", nameof(style));
        }

        var glyphs = GlyphPainter.Read(style.Options, Mix(seed, GlyphSeed));
        var ornaments = LetterOrnaments.Read(style.Options);
        var c = new Choices(style, seed);
        bool rtl = Direction(content);
        int width = style.Width;
        float margin = width * c.Between(style.Margin);
        float inner = width - 2 * margin;
        var bodyFont = c.Font();
        var headerFont = c.Chance(style.OtherHeaderFont) ? c.Font() : bodyFont;
        float bodySize = c.Between(style.BodySize), spacing = c.Between(style.LineSpacing);
        var pairs = Hand(seed);
        var body = Style(style, bodyFont, bodySize, spacing, content.Language, pairs, seed);
        var header = Style(style, headerFont, bodySize * c.Between(0.8f, 1.0f), c.Between(1.0f, 1.3f), content.Language, pairs, seed);
        var title = Style(style, headerFont, bodySize * c.Between(1.1f, 1.4f), 1.1f, content.Language, pairs, seed);
        var footer = Style(style, bodyFont, bodySize * c.Between(0.6f, 0.8f), 1.1f, content.Language, pairs, seed);
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

        // The pen-drawn signature over the name: its box and stroke from a seed of its own (every draw made, taken or not).
        int signing = Mix(seed, SignatureSeed);
        var signatureChoices = new Random(Mix(signing, 1));
        bool signed = content.Signature.Count > 0 && signatureChoices.NextDouble() < ornaments.Signature;
        float signedWidth = signatureWidth * (0.55f + 0.35f * (float)signatureChoices.NextDouble());
        float signedX = signatureX + (signatureWidth - signedWidth) * (0.25f + 0.5f * (float)signatureChoices.NextDouble());
        float signedHeight = bodySize * (1.5f + 1.1f * (float)signatureChoices.NextDouble());
        float signedStroke = Math.Max(1f, bodySize * (0.045f + 0.04f * (float)signatureChoices.NextDouble()));
        float signedGap = signedHeight * (0.85f + 0.15f * (float)signatureChoices.NextDouble());
        if (signed)
        {
            y += signedGap;
        }

        Add("signature", content.Signature, body, signatureX, signatureWidth, TextAlignment.Center, bodySize, blueSignature && !signed);
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
        var record = style.Vectors ? new PageRecord() : null;
        var blocks = new List<PageBlock>();
        foreach (var (kind, text, block, x, top, blue) in placed)
        {
            record?.Text(block, x, top, blue, glyphs, glyphs.Line);
            glyphs.Draw(block, x, top, blue ? canvas.Blue : canvas.Ink, width, height);
            blocks.Add(new PageBlock(kind, text, x, top, block.Width, block.Height));
        }

        // The ornaments' texts: the page's first header line (or its title), in the header's font.
        string? name = content.HeaderStart.FirstOrDefault(l => l.Trim().Length > 0) ?? content.Title;
        TextStyle Lettering(float size) => new(headerFont, size) { Fallbacks = style.Fallbacks, Language = content.Language };
        string? emblemName = null;
        if (emblem)
        {
            int e = Mix(seed, EmblemSeed);
            var profile = ornaments.Emblems[new Random(Mix(e, 1)).Next(ornaments.Emblems.Count)];
            float cx = width / 2f, cy = margin * 0.7f + emblemRadius + 2;
            var box = new ShapeContext
            {
                X = cx - emblemRadius, Y = cy - emblemRadius, Width = 2 * emblemRadius, Height = 2 * emblemRadius, RightToLeft = rtl,
                TextSize = header.Size, LineHeight = header.Size * 1.2f, StrokeWidth = Math.Max(1f, emblemRadius * 0.05f),
            };
            var (drawn, _) = ShapesLayout.Plan(Sample(profile, Mix(e, 2)), box, Mix(e, 3));
            var lettering = new ShapeLettering(Lettering(header.Size), name, rtl);
            record?.Shapes(drawn, box.StrokeWidth, lettering, blue: false);
            var ink = ShapeDrawing.Draw(canvas, canvas.Ink, drawn, ornaments.EmblemPen, ornaments.EmblemPenSettings, Mix(e, 4), box.StrokeWidth, null, lettering);
            blocks.Add(Ornament("emblem", box, ink, width, height));
            emblemName = Named(profile);
        }

        if (rule)
        {
            record?.Rule(margin, ruleY, inner, ruleThickness);
            canvas.Rectangle(canvas.Ink, margin, ruleY, inner, ruleThickness);
            blocks.Add(new PageBlock("rule", "", margin, ruleY, inner, ruleThickness));
        }

        string? signatureName = null;
        if (signed)
        {
            var profile = ornaments.Signatures[signatureChoices.Next(ornaments.Signatures.Count)];
            var box = new ShapeContext
            {
                X = signedX, Y = signatureTop, Width = signedWidth, Height = signedHeight, RightToLeft = rtl, TextSize = bodySize, LineHeight = bodySize * 1.2f, StrokeWidth = signedStroke,
            };
            var (drawn, _) = ShapesLayout.Plan(Sample(profile, Mix(signing, 2)), box, Mix(signing, 3), Scribble);
            var lettering = new ShapeLettering(Lettering(bodySize), name, rtl);
            record?.Shapes(drawn, signedStroke, lettering, blueSignature);
            var ink = ShapeDrawing.Draw(canvas, blueSignature ? canvas.Blue : canvas.Ink, drawn, ornaments.SignaturePen, ornaments.SignaturePenSettings, Mix(signing, 4), signedStroke,
                null, lettering);
            blocks.Add(Ornament("scribble", box, ink, width, height));
            signatureName = $"{Named(profile)} with {ornaments.SignaturePenName}";
        }

        string? stampName = null;
        if (stamp)
        {
            float cx = signatureX + signatureWidth * c.Between(0.2f, 0.8f), cy = signatureTop + c.Between(-0.5f, 1.0f) * stampRadius;
            int st = Mix(seed, StampSeed);
            var stampChoices = new Random(Mix(st, 1));
            var profile = ornaments.Stamps[stampChoices.Next(ornaments.Stamps.Count)];
            float turn = Between(stampChoices, ornaments.StampTurn) * MathF.PI / 180, uneven = Between(stampChoices, ornaments.StampInk);
            bool black = stampChoices.NextDouble() < ornaments.StampBlack;
            var layer = style.Color && !black ? canvas.Blue : canvas.Ink;
            float stroke = Math.Max(1.5f, stampRadius * 0.045f);
            var box = new ShapeContext
            {
                X = cx - stampRadius, Y = cy - stampRadius, Width = 2 * stampRadius, Height = 2 * stampRadius, RightToLeft = rtl, TextSize = bodySize, LineHeight = bodySize * 1.2f,
                StrokeWidth = stroke,
            };
            var (drawn, _) = ShapesLayout.Plan(Sample(profile, Mix(st, 2)), box, Mix(st, 3));

            // Drawn turned in a canvas of its own (its square turned, its widest stroke and the pen's reach inside it), then
            // laid on the page with a stamp's uneven ink.
            float reach = stampRadius * 1.415f + 3 * stroke + ornaments.StampPen.Stray(ornaments.StampPenSettings, 3 * stroke) + 2;
            int left = (int)MathF.Floor(cx - reach), top = (int)MathF.Floor(cy - reach), extent = (int)MathF.Ceiling(2 * reach) + 1;
            var lettering = new ShapeLettering(Lettering(bodySize), name, rtl);
            record?.Shapes(ShapeDrawing.Turned(drawn, cx, cy, turn, 0, 0), stroke, lettering, blue: style.Color && !black);
            using (var own = new PageCanvas(extent, extent))
            {
                var ink = ShapeDrawing.Draw(own, own.Ink, ShapeDrawing.Turned(drawn, cx, cy, turn, -left, -top), ornaments.StampPen, ornaments.StampPenSettings, Mix(st, 4), stroke, null,
                    lettering);
                ShapeDrawing.LayUneven(layer, width, height, own.Ink, left, top, extent, extent,
                    new UnevenInk(Mix(st, 5), uneven, extent, extent, stampRadius * 0.35f, Math.Max(1.5f, stampRadius * 0.06f)));
                blocks.Add(Ornament("stamp", box, (ink.X0 + left, ink.Y0 + top, ink.X1 + left, ink.Y1 + top), width, height));
            }

            stampName = Named(profile) + (style.Color && black ? " black" : "");
        }

        if (footerBlock is not null)
        {
            float footerTop = height - margin * 0.6f - footerBlock.Height;
            record?.Rule(margin, footerTop - bodySize * 0.5f, inner, 1.5f);
            canvas.Rectangle(canvas.Ink, margin, footerTop - bodySize * 0.5f, inner, 1.5f);
            record?.Text(footerBlock, margin, footerTop, false, glyphs, glyphs.Line);
            glyphs.Draw(footerBlock, margin, footerTop, canvas.Ink, width, height);
            blocks.Add(new PageBlock("footer", string.Join("\n", content.Footer), margin, footerTop, inner, footerBlock.Height));
        }

        // The page's colours drawn in the order Compose takes them (its arguments, left to right).
        var paper = Paper(c, style.Color);
        float darkness = c.Between(style.Ink);
        var blueInk = BlueInk(c);
        var image = canvas.Compose(paper, darkness, blueInk, style.Color);
        string description = string.Create(CultureInfo.InvariantCulture,
            $"letter: body {Describe(body)}, header {Describe(header)}, spacing {spacing:F2}, margin {margin:F0}px, {(justify ? "justified" : "start")}{(emblem ? $", emblem {emblemName}" : "")}{(rule ? ", rule" : "")}{(stamp ? $", stamp {stampName}" : "")}{(blueSignature ? ", blue signature" : "")}{(signed ? $", signed {signatureName}" : "")}{Paired(body)}{Posed(glyphs)}");
        return new RenderedPage(image, content.ReadingOrder(), blocks, description) { Vectors = record?.Finish(width, height, paper, darkness, blueInk, style.Color) };
    }

    // A profile's name in a description ("written" for one written as text).
    private static string Named(string profile) => profile == "custom" ? "written" : profile;

    // A shape profile's steps for a seed: sampling the profiles by name is not on the new design yet. The default signature,
    // "scribble" (one step, drawn with the scribble shape: Scribble), is reached directly; any other profile (every emblem and
    // stamp: no one of them is the default) stops.
    private static readonly IReadOnlyList<ShapeStep> ScribbleSteps = [new ShapeStep(Ornaments.Signature)];

    private static readonly ScribbleShape Scribble = new();

    private static IReadOnlyList<ShapeStep> Sample(string profile, int seed) =>
        profile.Equals(Ornaments.Signature, StringComparison.OrdinalIgnoreCase) ? ScribbleSteps
        : throw new NotSupportedException($"The shape profile '{profile}' by name is not on the new design yet.");

    private static float Between(Random random, (float Min, float Max) range) => range.Min + (float)random.NextDouble() * (range.Max - range.Min);

    // An ornament's block: its box and what it drew, inside the page.
    private static PageBlock Ornament(string kind, ShapeContext box, (float X0, float Y0, float X1, float Y1) ink, int width, int height)
    {
        float x0 = Math.Max(0, Math.Min(box.X, ink.X0)), y0 = Math.Max(0, Math.Min(box.Y, ink.Y0));
        float x1 = Math.Min(width, Math.Max(box.Right, ink.X1)), y1 = Math.Min(height, Math.Max(box.Bottom, ink.Y1));
        return new PageBlock(kind, "", x0, y0, Math.Max(0, x1 - x0), Math.Max(0, y1 - y0));
    }
}

/// <summary>
/// A letter's ornament settings, read once per page from its options (<see cref="PageStyle.Options"/>): the emblems,
/// stamps and signatures it draws one of (<c>emblem-shapes</c>, <c>stamp-shapes</c>, <c>signature-shapes</c>: shape
/// profiles between '|'), their pens (<c>emblem-pen</c> and <c>stamp-pen</c>, ideal; <c>signature-pen</c>, ballpoint;
/// each with the pens' own keys), how far a stamp is turned (<c>stamp-turn</c>, degrees, -12..12), how uneven its ink is
/// (<c>stamp-ink</c>, 0 to 1, 0.2..0.65), the chance a colour page's stamp is black (<c>stamp-black</c>, 0.2), and the
/// chance a signature block has a signature drawn over its name (<c>signature</c>, 1).
/// </summary>
public sealed class LetterOrnaments
{
    /// <summary>The emblems' shape profiles (<c>emblem-shapes</c>), one drawn per page.</summary>
    public required IReadOnlyList<string> Emblems { get; init; }

    /// <summary>The stamps' shape profiles (<c>stamp-shapes</c>), one drawn per page.</summary>
    public required IReadOnlyList<string> Stamps { get; init; }

    /// <summary>The signatures' shape profiles (<c>signature-shapes</c>), one drawn per page.</summary>
    public required IReadOnlyList<string> Signatures { get; init; }

    /// <summary>The emblem's pen (<c>emblem-pen</c>).</summary>
    public required IPen EmblemPen { get; init; }

    /// <summary>The emblem pen's settings, its own keys of the options.</summary>
    public required PenSettings EmblemPenSettings { get; init; }

    /// <summary>The stamp's pen (<c>stamp-pen</c>).</summary>
    public required IPen StampPen { get; init; }

    /// <summary>The stamp pen's settings, its own keys of the options.</summary>
    public required PenSettings StampPenSettings { get; init; }

    /// <summary>The signature's pen (<c>signature-pen</c>).</summary>
    public required IPen SignaturePen { get; init; }

    /// <summary>The signature pen's name, as the options asked for it.</summary>
    public required string SignaturePenName { get; init; }

    /// <summary>The signature pen's settings, its own keys of the options.</summary>
    public required PenSettings SignaturePenSettings { get; init; }

    /// <summary>How far a stamp is turned, in degrees (<c>stamp-turn</c>).</summary>
    public required (float Min, float Max) StampTurn { get; init; }

    /// <summary>How uneven a stamp's ink is, 0 to 1 (<c>stamp-ink</c>).</summary>
    public required (float Min, float Max) StampInk { get; init; }

    /// <summary>The chance a colour page's stamp is black (<c>stamp-black</c>).</summary>
    public required float StampBlack { get; init; }

    /// <summary>The chance a signature block has a signature drawn over its name (<c>signature</c>).</summary>
    public required float Signature { get; init; }

    // The signature's default pen, the ballpoint, reached directly with its own keys of the page's options.
    private static readonly BallpointPen Ballpoint = new();

    // The shape profiles between '|' (a name given twice is drawn twice as often). Checking each by its name is not on
    // the new design yet: a profile is looked up when the letter draws it
    // (LetterLayout's Sample: the default signature reached directly, any other stopping there).
    private static IReadOnlyList<string> Choices(string list)
    {
        var items = list.Split('|', StringSplitOptions.TrimEntries);
        if (items.Any(i => i.Length == 0))
        {
            throw new ArgumentException($"Shape profiles are names between '|', none empty, not '{list}'.", nameof(list));
        }

        return items;
    }

    /// <summary>The settings <paramref name="options"/> give, the defaults for those not given.</summary>
    /// <exception cref="ArgumentException">A bad value; the message names the setting.</exception>
    /// <exception cref="NotSupportedException">A pen other than the defaults (ideal; the signature's ballpoint): a pen by name is not on the new design yet.</exception>
    public static LetterOrnaments Read(IReadOnlyDictionary<string, string> options)
    {
        string? Given(string key) => options.TryGetValue(key, out var value) && value.Length > 0 ? value : null;
        (IPen, PenSettings) Pen(string key, string otherwise) => ShapesLayout.PenFor(options, Given(key) ?? otherwise);

        var (emblemPen, emblemSettings) = Pen("emblem-pen", Pens.Ideal);
        var (stampPen, stampSettings) = Pen("stamp-pen", Pens.Ideal);
        string signaturePenName = Given("signature-pen") ?? Ballpoint.Name;
        var (signaturePen, signatureSettings) = signaturePenName.Equals(Ballpoint.Name, StringComparison.OrdinalIgnoreCase)
            ? (Ballpoint, Ballpoint.Read(PageOptions.Only(options, BallpointPen.OwnKeys)))
            : ShapesLayout.PenFor(options, signaturePenName);
        return new LetterOrnaments
        {
            Emblems = Choices(Given("emblem-shapes") ?? Ornaments.Emblems),
            Stamps = Choices(Given("stamp-shapes") ?? Ornaments.Stamps),
            Signatures = Choices(Given("signature-shapes") ?? Ornaments.Signature),
            EmblemPen = emblemPen, EmblemPenSettings = emblemSettings, StampPen = stampPen, StampPenSettings = stampSettings, SignaturePen = signaturePen,
            SignaturePenName = signaturePenName, SignaturePenSettings = signatureSettings,
            StampTurn = Given("stamp-turn") is { } turn ? PageOptions.Range("Page setting stamp-turn", turn, -180, 180) : (-12, 12),
            StampInk = Given("stamp-ink") is { } ink ? PageOptions.Range("Page setting stamp-ink", ink, 0, 1) : (0.2f, 0.65f),
            StampBlack = Given("stamp-black") is { } black ? PageOptions.Chance("Page setting stamp-black", black) : 0.2f,
            Signature = Given("signature") is { } signature ? PageOptions.Chance("Page setting signature", signature) : 1,
        };
    }
}

/// <summary>"plain": header lines at the start, a centred title, the subject, the body, centred footer lines; no rule or ornament.</summary>
public sealed class PlainLayout : IPageLayout
{
    /// <summary>"plain".</summary>
    public string Name => "plain";

    /// <summary>One line on what it draws.</summary>
    public string Summary => "a plain page: header lines, title, subject, body, footer; no rule or ornament";

    /// <inheritdoc/>
    public RenderedPage Render(PageContent content, PageStyle style, int seed)
    {
        ArgumentNullException.ThrowIfNull(content);
        ArgumentNullException.ThrowIfNull(style);
        if (style.Fonts.Count == 0)
        {
            throw new ArgumentException("A page needs at least one font.", nameof(style));
        }

        var glyphs = GlyphPainter.Read(style.Options, Mix(seed, GlyphSeed));
        var c = new Choices(style, seed);
        bool rtl = Direction(content);
        int width = style.Width;
        float margin = width * c.Between(style.Margin), inner = width - 2 * margin;
        var font = c.Font();
        float size = c.Between(style.BodySize), spacing = c.Between(style.LineSpacing);
        var body = Style(style, font, size, spacing, content.Language, Hand(seed), seed);
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
        var record = style.Vectors ? new PageRecord() : null;
        var blocks = new List<PageBlock>();
        foreach (var (kind, text, block, top) in placed)
        {
            record?.Text(block, margin, top, false, glyphs, glyphs.Line);
            glyphs.Draw(block, margin, top, canvas.Ink, width, height);
            blocks.Add(new PageBlock(kind, text, margin, top, inner, block.Height));
        }

        // The page's colours drawn in the order Compose takes them (its arguments, left to right).
        var paper = Paper(c, style.Color);
        float darkness = c.Between(style.Ink);
        var blueInk = BlueInk(c);
        var image = canvas.Compose(paper, darkness, blueInk, style.Color);
        return new RenderedPage(image, content.ReadingOrder(), blocks,
            string.Create(CultureInfo.InvariantCulture, $"plain: {Describe(body)}, spacing {spacing:F2}, margin {margin:F0}px, {(justify ? "justified" : "start")}{Paired(body)}{Posed(glyphs)}"))
        {
            Vectors = record?.Finish(width, height, paper, darkness, blueInk, style.Color),
        };
    }
}
