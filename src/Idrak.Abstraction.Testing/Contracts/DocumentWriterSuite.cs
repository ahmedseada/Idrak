// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Data;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// The checks of a document writer (<see cref="IDocumentWriter"/>, registered in <see cref="DocumentWriters"/>): pages of
/// each kind (images stored exact, vector paths under a clip, glyph runs of <see cref="Face"/>, hidden text, information
/// and an attached file) written, then read back through <see cref="Reader"/>:
/// <list type="bullet">
/// <item>the same input writes the same bytes;</item>
/// <item>the pages' count and sizes, every path (verbs, points, fill, stroke, dash, colours, clip), every image's raw
/// pixels, the text and glyphs of the runs (hidden ones hidden), the information and the attached file's bytes read back
/// as written;</item>
/// <item>a page's memory is bounded: writing more pages keeps no more memory than a few pages' images.</item>
/// </list>
/// Points and sizes may differ by a thousandth of a point, glyph places by a hundredth, pixels by float rounding.
/// </summary>
/// <param name="reader">The format the written files are read back with.</param>
/// <param name="face">A font whose glyphs the text pages are written in; null leaves text out (its checks are skipped).</param>
/// <param name="options">The options documents are written with; null for the defaults.</param>
public sealed class DocumentWriterSuite(IDocumentFormat reader, IFontFace? face = null, DocumentWriteOptions? options = null) : ContractSuite<IDocumentWriter>
{
    private const float PointTolerance = 1e-3f, GlyphTolerance = 1e-2f, PixelTolerance = 1e-5f;

    private readonly DocumentWriteOptions _options = options ?? new DocumentWriteOptions();

    /// <summary>The format the written files are read back with.</summary>
    public IDocumentFormat Reader { get; } = reader ?? throw new ArgumentNullException(nameof(reader));

    /// <summary>The font of the text pages, or null.</summary>
    public IFontFace? Face { get; } = face;

    /// <inheritdoc />
    public override string Name => "document-writer";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        yield return Case("one blank page", "blank", 1, 1);
        yield return Case("a page of images, grey and colour, stored exact", "images", 2, 1);
        yield return Case("a page of paths: lines and curves, both fill rules, strokes with caps, joins and dashes, colours, a clip", "paths", 3, 1);
        yield return Case("a page of text: glyphs standing for their text, a hidden line", "text", 4, 1);
        yield return Case("pages of every kind, with information and an attached file", "mixed", 5, 3);
        yield return Case("a page's memory is bounded: more pages keep no more", "memory", 6, 8);
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        int pages = random.Next(1, large ? 40 : 5);
        return Case($"{pages} random pages", "mixed", random.Next(), pages);
    }

    /// <inheritdoc />
    public override void Run(IDocumentWriter implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        string kind = (string)@case.Data["kind"]!;
        int seed = (int)@case.Data["seed"]!, pages = (int)@case.Data["pages"]!;
        if (kind == "memory")
        {
            RunMemory(implementation, pages, checks);
            return;
        }

        if (kind == "text" && Face is null)
        {
            checks.Skip("the runs' text and glyphs", "no font given");
            return;
        }

        var plan = Plan(kind, new Random(seed), pages);
        byte[] first = Write(implementation, plan), second = Write(implementation, plan);
        checks.Check("the same input writes the same bytes", first.AsSpan().SequenceEqual(second), $"{first.Length} and {second.Length} bytes differ");

        using var document = Reader.Open(new MemoryStream(first, writable: false), "written", new DocumentOptions());
        checks.Check("the page count", document.PageCount == plan.Pages.Count, $"{document.PageCount} pages read, {plan.Pages.Count} written");
        for (int i = 0; i < Math.Min(document.PageCount, plan.Pages.Count); i++)
        {
            ComparePage(plan.Pages[i], document.ReadPage(i, DocumentParts.All), checks);
        }

        if (plan.Info != DocumentInfo.Empty)
        {
            var info = document.Info;
            checks.Compare("the information", Comparisons.Exact((plan.Info.Title, plan.Info.Author, plan.Info.Subject, plan.Info.Keywords, plan.Info.Creator),
                (info.Title, info.Author, info.Subject, info.Keywords, info.Creator)));
        }

        foreach (var (name, mediaType, bytes) in plan.Attachments)
        {
            var attachment = document.Attachments.FirstOrDefault(a => a.Name == name);
            if (attachment is null)
            {
                checks.Fail("the attached file's bytes", $"no attachment {name} (read: {string.Join(", ", document.Attachments.Select(a => a.Name))})");
                continue;
            }

            using var content = attachment.Read();
            using var copy = new MemoryStream();
            content.CopyTo(copy);
            checks.Check("the attached file's bytes", copy.ToArray().AsSpan().SequenceEqual(bytes) && (attachment.MediaType is null || attachment.MediaType == mediaType),
                $"{name}: {copy.Length} bytes of {bytes.Length}, media type {attachment.MediaType}");
        }
    }

    private void ComparePage(PagePlan plan, DocumentPage page, CaseChecks checks)
    {
        checks.Check("each page's size", MathF.Abs(page.Width - plan.Width) <= PointTolerance && MathF.Abs(page.Height - plan.Height) <= PointTolerance,
            $"page {page.Index}: {page.Width} x {page.Height}, written {plan.Width} x {plan.Height}");

        var paths = plan.Parts.OfType<PathPart>().ToList();
        checks.Check("every path", page.Paths.Count == paths.Count, $"page {page.Index}: {page.Paths.Count} paths read, {paths.Count} written");
        for (int k = 0; k < Math.Min(paths.Count, page.Paths.Count); k++)
        {
            checks.Compare("every path", ComparePath(paths[k].Path, page.Paths[k], paths[k].Clipped ? 0 : -1));
        }

        var clips = plan.Parts.OfType<ClipPart>().ToList();
        if (clips.Count > 0)
        {
            checks.Check("the clip", page.Clips.Count == clips.Count && ComparePath(clips[0].Path, page.Clips[0], -1, clip: true) is null,
                $"page {page.Index}: {page.Clips.Count} clips read, {clips.Count} written{(page.Clips.Count > 0 ? ": " + ComparePath(clips[0].Path, page.Clips[0], -1, clip: true) : "")}");
        }

        var images = plan.Parts.OfType<ImagePart>().ToList();
        checks.Check("every image's pixels", page.Images.Count == images.Count, $"page {page.Index}: {page.Images.Count} images read, {images.Count} written");
        for (int k = 0; k < Math.Min(images.Count, page.Images.Count); k++)
        {
            var (expected, stored) = (images[k], page.Images[k]);
            var pixels = stored.Decode();
            string? box = BoxDifference(expected.Box, stored.Box);
            checks.Compare("every image's pixels", box is not null ? $"page {page.Index}, image {k}: {box}"
                : Comparisons.Exact((expected.Image.Channels, expected.Image.Height, expected.Image.Width), (pixels.Channels, pixels.Height, pixels.Width))
                  ?? Comparisons.Difference(expected.Image.Pixels, pixels.Pixels, PixelTolerance));
        }

        var texts = plan.Parts.OfType<TextPart>().ToList();
        if (texts.Count == 0)
        {
            return;
        }

        var runs = page.Runs.OrderBy(r => r.Order).ToList();
        checks.Compare("the runs' text", Comparisons.Difference(string.Concat(texts.Select(t => t.Text)), string.Concat(runs.Select(r => r.Text))));
        checks.Compare("hidden text read back hidden", Comparisons.Difference(string.Concat(texts.Where(t => t.Mode == DocumentTextMode.Hidden).Select(t => t.Text)),
            string.Concat(runs.Where(r => r.Hidden).Select(r => r.Text))));
        var written = texts.SelectMany(t => t.Glyphs).ToArray();
        var read = runs.SelectMany(r => r.Glyphs.ToArray()).ToArray();
        int wrong = Enumerable.Range(0, Math.Min(written.Length, read.Length))
            .FirstOrDefault(g => written[g].Glyph != read[g].Glyph || MathF.Abs(written[g].X - read[g].X) > GlyphTolerance || MathF.Abs(written[g].Y - read[g].Y) > GlyphTolerance, -1);
        checks.Check("the runs' glyphs and their places", written.Length == read.Length && wrong < 0,
            wrong < 0 ? $"{read.Length} glyphs read, {written.Length} written" : $"glyph {wrong} reads as {read[wrong]}, written {written[wrong]}");
    }

    private static string? ComparePath(DocumentPath expected, DocumentPath actual, int clipIndex, bool clip = false)
    {
        if (!expected.Verbs.Span.SequenceEqual(actual.Verbs.Span))
        {
            return $"verbs {string.Join(" ", actual.Verbs.ToArray())}, written {string.Join(" ", expected.Verbs.ToArray())}";
        }

        string? points = Comparisons.Difference(expected.Points.Span, actual.Points.Span, PointTolerance);
        if (points is not null)
        {
            return $"points: {points}";
        }

        if (expected.Fill != actual.Fill)
        {
            return $"fill {actual.Fill}, written {expected.Fill}";
        }

        if (clip)
        {
            return null;
        }

        if (expected.Stroke.HasValue != actual.Stroke.HasValue || (expected.Stroke is { } w && MathF.Abs(w - actual.Stroke!.Value) > PointTolerance))
        {
            return $"stroke {actual.Stroke}, written {expected.Stroke}";
        }

        if (expected.Stroke is not null)
        {
            if (expected.Cap != actual.Cap || expected.Join != actual.Join || MathF.Abs(expected.MiterLimit - actual.MiterLimit) > PointTolerance)
            {
                return $"cap, join and miter {actual.Cap} {actual.Join} {actual.MiterLimit}, written {expected.Cap} {expected.Join} {expected.MiterLimit}";
            }

            if (Comparisons.Difference(expected.Dash.Span, actual.Dash.Span, PointTolerance) is not null || MathF.Abs(expected.DashPhase - actual.DashPhase) > PointTolerance)
            {
                return $"dash {string.Join(" ", actual.Dash.ToArray())} from {actual.DashPhase}, written {string.Join(" ", expected.Dash.ToArray())} from {expected.DashPhase}";
            }

            if (ColourDifference(expected.StrokeColour, actual.StrokeColour) is { } stroke)
            {
                return $"stroke colour: {stroke}";
            }
        }

        if (expected.Fill != DocumentFill.None && ColourDifference(expected.FillColour, actual.FillColour) is { } fill)
        {
            return $"fill colour: {fill}";
        }

        return actual.Clip == clipIndex ? null : $"drawn under clip {actual.Clip}, written under {clipIndex}";
    }

    private static string? ColourDifference(DocumentColour expected, DocumentColour actual)
    {
        if (expected.Space != actual.Space)
        {
            return $"{actual}, written {expected}";
        }

        for (int c = 0; c < expected.Count; c++)
        {
            if (MathF.Abs(expected[c] - actual[c]) > PointTolerance)
            {
                return $"{actual}, written {expected}";
            }
        }

        return MathF.Abs(expected.Alpha - actual.Alpha) > PointTolerance ? $"{actual}, written {expected}" : null;
    }

    private static string? BoxDifference(DocumentBox expected, DocumentBox actual) =>
        MathF.Abs(expected.X - actual.X) <= PointTolerance && MathF.Abs(expected.Y - actual.Y) <= PointTolerance
        && MathF.Abs(expected.Width - actual.Width) <= PointTolerance && MathF.Abs(expected.Height - actual.Height) <= PointTolerance
            ? null
            : $"box {actual}, written {expected}";

    // Pages of the same noise image written to nothing: what the process keeps after the first pages is measured, and after
    // more; the difference stays under a few pages' images (a writer that kept every page would keep them all).
    private void RunMemory(IDocumentWriter implementation, int pages, CaseChecks checks)
    {
        var random = new Random(6);
        const int Side = 256;
        var pixels = new float[3 * Side * Side];
        for (int i = 0; i < pixels.Length; i++)
        {
            pixels[i] = random.Next(256) / 255f;
        }

        var image = new ImageData(pixels, 3, Side, Side);
        var box = new DocumentBox(36f, 36f, Side, Side);
        using var builder = implementation.Create(Stream.Null, _options with { ImageEncoding = DocumentImageEncoding.Flate });
        long before = 0;
        for (int page = 0; page < pages; page++)
        {
            builder.AddPage(Side + 72f, Side + 72f).Image(image, box, DocumentImageEncoding.Flate);
            if (page == 1)
            {
                before = GC.GetTotalMemory(forceFullCollection: true);
            }
        }

        builder.AddPage(72f, 72f);                                             // the last image page is written
        long after = GC.GetTotalMemory(forceFullCollection: true);
        long bound = 3L * pixels.Length;                                       // three pages' images at a byte a value
        checks.Check("a page's memory is bounded", after - before <= bound,
            $"{pages} pages of a {Side} x {Side} image: {after - before:N0} bytes more after page 2, more than {bound:N0}");
        builder.Finish();
    }

    private byte[] Write(IDocumentWriter implementation, DocumentPlan plan)
    {
        using var destination = new MemoryStream();
        using (var builder = implementation.Create(destination, _options))
        {
            if (plan.Info != DocumentInfo.Empty)
            {
                builder.Info = plan.Info;
            }

            foreach (var page in plan.Pages)
            {
                var writer = builder.AddPage(page.Width, page.Height);
                foreach (var part in page.Parts)
                {
                    switch (part)
                    {
                        case ImagePart image:
                            writer.Image(image.Image, image.Box, DocumentImageEncoding.Flate);
                            break;
                        case PathPart path:
                            writer.Path(path.Path);
                            break;
                        case ClipPart clip:
                            writer.Save();
                            writer.Clip(clip.Path);
                            break;
                        case RestorePart:
                            writer.Restore();
                            break;
                        case TextPart text:
                            writer.Glyphs(Face!, text.Size, text.Glyphs, text.Text, text.Mode, text.Colour);
                            break;
                    }
                }
            }

            foreach (var (name, mediaType, bytes) in plan.Attachments)
            {
                builder.Attach(name, mediaType, new MemoryStream(bytes, writable: false));
            }

            builder.Finish();
        }

        return destination.ToArray();
    }

    // The document a case writes: the same for the same kind and seed.
    private DocumentPlan Plan(string kind, Random random, int pageCount)
    {
        var plan = new DocumentPlan();
        switch (kind)
        {
            case "blank":
                plan.Pages.Add(new PagePlan(612f, 792f));
                break;
            case "images":
                plan.Pages.Add(ImagePage(random));
                break;
            case "paths":
                plan.Pages.Add(PathPage(random));
                break;
            case "text":
                plan.Pages.Add(TextPage(random));
                break;
            default:
                for (int p = 0; p < pageCount; p++)
                {
                    plan.Pages.Add((p % 3) switch { 0 => ImagePage(random), 1 => PathPage(random), _ => Face is null ? PathPage(random) : TextPage(random) });
                }

                plan.Info = new DocumentInfo { Title = "A technical report", Author = "The writer suite", Subject = "Quarterly results", Keywords = "report, results", Creator = "Idrak.Abstraction.Testing" };
                plan.Attachments.Add(("notes.txt", "text/plain", Encoding.UTF8.GetBytes("Notes for the report: the figures are final.\n")));
                break;
        }

        return plan;
    }

    private static PagePlan ImagePage(Random random)
    {
        var page = new PagePlan(400f + random.Next(200), 500f + random.Next(300));
        foreach (int channels in (int[])[1, 3])
        {
            int width = 1 + random.Next(24), height = 1 + random.Next(24);
            var pixels = new float[channels * width * height];
            for (int i = 0; i < pixels.Length; i++)
            {
                pixels[i] = random.Next(256) / 255f;
            }

            float x = 20f + random.Next(150), y = 20f + random.Next(200);
            page.Parts.Add(new ImagePart(new ImageData(pixels, channels, height, width), new DocumentBox(x, y, width * 4f, height * 4f)));
        }

        return page;
    }

    private static PagePlan PathPage(Random random)
    {
        var page = new PagePlan(500f, 600f);
        float P() => MathF.Round(10f + random.NextSingle() * 400f, 2);
        page.Parts.Add(new PathPart(new DocumentPath([OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close], [P(), P(), P(), P(), P(), P()])
        {
            Fill = DocumentFill.NonZero, FillColour = DocumentColour.Rgb(random.Next(256) / 255f, random.Next(256) / 255f, random.Next(256) / 255f),
        }));
        page.Parts.Add(new PathPart(new DocumentPath([OutlineVerb.Move, OutlineVerb.Quadratic, OutlineVerb.Cubic, OutlineVerb.Close], [P(), P(), P(), P(), P(), P(), P(), P(), P(), P(), P(), P()])
        {
            Fill = DocumentFill.EvenOdd, FillColour = DocumentColour.Grey(random.Next(256) / 255f), Stroke = 1.5f, Cap = DocumentLineCap.Round, Join = DocumentLineJoin.Bevel,
            StrokeColour = DocumentColour.Cmyk(0.1f, 0.2f, 0.3f, 0.4f),
        }));
        page.Parts.Add(new ClipPart(DocumentPath.Rectangle(new DocumentBox(50f, 50f, 300f, 300f))));
        page.Parts.Add(new PathPart(new DocumentPath([OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line], [P(), P(), P(), P(), P(), P()])
        {
            Fill = DocumentFill.None, Stroke = 2.25f, Cap = DocumentLineCap.Square, Join = DocumentLineJoin.Miter, MiterLimit = 4f, Dash = new float[] { 6f, 3f }, DashPhase = 1f,
            StrokeColour = DocumentColour.Rgb(0.2f, 0.4f, 0.6f),
        }, Clipped: true));
        page.Parts.Add(new RestorePart());
        page.Parts.Add(new PathPart(DocumentPath.Rectangle(new DocumentBox(P(), P(), 30f, 20f))));
        return page;
    }

    private PagePlan TextPage(Random random)
    {
        var page = new PagePlan(612f, 792f);
        var face = Face!;
        float size = 10f + random.Next(14), y = 100f;
        foreach (var (text, mode) in new[] { ("A plain line of text, number 42.", DocumentTextMode.Fill), ("Thank you for your letter.", DocumentTextMode.Fill), ("A hidden line over a scan.", DocumentTextMode.Hidden) })
        {
            var glyphs = new List<DocumentGlyph>();
            var kept = new StringBuilder();
            float x = 72f, scale = size / face.Metrics.UnitsPerEm;
            foreach (var rune in text.EnumerateRunes())
            {
                int glyph = face.Glyph(rune.Value);
                if (glyph == 0)
                {
                    continue;
                }

                float advance = face.Advance(glyph) * scale;
                glyphs.Add(new DocumentGlyph(glyph, MathF.Round(x, 2), y, advance));
                kept.Append(rune.ToString());
                x += advance;
            }

            if (glyphs.Count > 0)
            {
                page.Parts.Add(new TextPart([.. glyphs], kept.ToString(), size, mode, DocumentColour.Grey(random.Next(128) / 255f)));
            }

            y += size * 1.5f;
        }

        return page;
    }

    private static ContractCase Case(string name, string kind, int seed, int pages) => new(name, new JsonObject { ["kind"] = kind, ["seed"] = seed, ["pages"] = pages });

    private sealed class DocumentPlan
    {
        public List<PagePlan> Pages { get; } = [];

        public DocumentInfo Info { get; set; } = DocumentInfo.Empty;

        public List<(string Name, string MediaType, byte[] Bytes)> Attachments { get; } = [];
    }

    private sealed record PagePlan(float Width, float Height)
    {
        public List<object> Parts { get; } = [];
    }

    private sealed record ImagePart(ImageData Image, DocumentBox Box);

    private sealed record PathPart(DocumentPath Path, bool Clipped = false);

    private sealed record ClipPart(DocumentPath Path);

    private sealed record RestorePart;

    private sealed record TextPart(DocumentGlyph[] Glyphs, string Text, float Size, DocumentTextMode Mode, DocumentColour Colour);
}
