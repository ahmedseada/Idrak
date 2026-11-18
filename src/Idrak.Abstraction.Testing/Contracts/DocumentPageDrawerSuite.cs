// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Data;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// The checks of a page drawer (<see cref="IDocumentPageDrawer"/>, registered in <see cref="DocumentPageDrawers"/>) on
/// pages the suite builds, of known paths, glyph runs and images:
/// <list type="bullet">
/// <item>the image is the page's size times the pixels per point, grey or RGB as asked, the background where nothing is
/// drawn;</item>
/// <item>a filled square and a glyph run match the registered rasterizer's own coverage of the same outlines (the mean
/// difference within <see cref="Tolerance"/>); an even-odd ring leaves its hole, a nonzero one fills it; a stroke inks
/// its line and nothing far from it; an image fills its box with its colour;</item>
/// <item>parts left out when asked (paths, images, text), hidden text drawn only when asked;</item>
/// <item>a run whose font is not there is left out and counted in the issues, never thrown;</item>
/// <item>the same drawing again and from several threads.</item>
/// </list>
/// The random cases draw rectangles of random colours on a grid: each rectangle's middle has its colour.
/// </summary>
/// <param name="face">A font for the glyph runs; null leaves them out (their checks are skipped).</param>
/// <param name="tolerance">How far, on average, a drawing may be from the rasterizer's coverage (0.02).</param>
public sealed class DocumentPageDrawerSuite(IFontFace? face = null, float tolerance = 0.02f) : ContractSuite<IDocumentPageDrawer>
{
    private const float Width = 120f, Height = 90f, Ink = 2f / 255f;

    /// <summary>The font of the glyph runs, or null.</summary>
    public IFontFace? Face { get; } = face;

    /// <summary>How far, on average, a drawing may be from the rasterizer's coverage.</summary>
    public float Tolerance { get; } = tolerance;

    /// <inheritdoc />
    public override string Name => "document-page-drawer";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        foreach (string kind in (string[])["blank", "square", "ring", "stroke", "image", "glyphs", "missing-font", "parts", "hidden", "grey"])
        {
            yield return new ContractCase(kind switch
            {
                "blank" => "a blank page: the background, at the page's size",
                "square" => "a filled square: the rasterizer's coverage",
                "ring" => "rings: even-odd leaves the hole, nonzero fills it",
                "stroke" => "a stroked line: inked along it, nothing far from it",
                "image" => "an image in a box: its colour",
                "glyphs" => "a glyph run: the rasterizer's coverage of its outlines",
                "missing-font" => "a run whose font is not there: left out and counted",
                "parts" => "parts left out when asked",
                "hidden" => "hidden text drawn only when asked",
                _ => "grey pixels",
            }, new JsonObject { ["kind"] = kind, ["seed"] = 0 });
        }
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        return new ContractCase("random rectangles on a grid", new JsonObject { ["kind"] = "rectangles", ["seed"] = random.Next(), ["large"] = large });
    }

    /// <inheritdoc />
    public override void Run(IDocumentPageDrawer implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var options = new DocumentDrawOptions { PixelsPerPoint = 2f };
        switch ((string)@case.Data["kind"]!)
        {
            case "blank":
            {
                var page = new DocumentPage(0, Width, Height);
                var drawing = implementation.Draw(page, options);
                var (w, h) = DocumentPageDrawers.PixelSize(page, options);
                checks.Check("the page's size in pixels, RGB", drawing.Image.Width == w && drawing.Image.Height == h && drawing.Image.Channels == 3,
                    $"{drawing.Image.Channels} x {drawing.Image.Height} x {drawing.Image.Width}, expected 3 x {h} x {w}");
                checks.Check("the background where nothing is drawn", drawing.Image.Pixels.All(v => v >= 1f - Ink), "a blank page has pixels that are not white");
                var again = implementation.Draw(page, options with { Background = DocumentColour.Rgb(0f, 0f, 1f) });
                checks.Check("the background is the options'", Close(again.Image, 0, 1, 1, 0f) && Close(again.Image, 2, 1, 1, 1f), "a blue background is not blue");
                break;
            }

            case "square":
            {
                var square = DocumentPath.Rectangle(new DocumentBox(20.25f, 10.5f, 40f, 30f));
                var page = new DocumentPage(0, Width, Height) { Paths = [square] };
                var drawing = Same(implementation, page, options, checks);
                CheckCoverage(drawing.Image, [(Outline(square), options.PixelsPerPoint, 0f, 0f)], options, checks, "a filled square matches the rasterizer's coverage");
                break;
            }

            case "ring":
            {
                // An outer square and an inner one wound the same way: the inner is a hole only by the even-odd rule.
                var verbs = new[] { OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close, OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close };
                float[] points = [10f, 10f, 70f, 10f, 70f, 70f, 10f, 70f, 30f, 30f, 50f, 30f, 50f, 50f, 30f, 50f];
                var evenOdd = implementation.Draw(new DocumentPage(0, Width, Height) { Paths = [new DocumentPath(verbs, points) { Fill = DocumentFill.EvenOdd }] }, options).Image;
                var nonZero = implementation.Draw(new DocumentPage(0, Width, Height) { Paths = [new DocumentPath(verbs, points) { Fill = DocumentFill.NonZero }] }, options).Image;
                checks.Check("even-odd leaves the hole", Grey(evenOdd, 40f, 40f) >= 1f - Ink && Grey(evenOdd, 20f, 20f) <= Ink,
                    $"the hole is {Grey(evenOdd, 40f, 40f)}, the ring {Grey(evenOdd, 20f, 20f)}");
                checks.Check("nonzero fills it", Grey(nonZero, 40f, 40f) <= Ink && Grey(nonZero, 20f, 20f) <= Ink, $"the middle is {Grey(nonZero, 40f, 40f)}, the ring {Grey(nonZero, 20f, 20f)}");
                break;
            }

            case "stroke":
            {
                var line = new DocumentPath([OutlineVerb.Move, OutlineVerb.Line], [10f, 45f, 110f, 45f]) { Fill = DocumentFill.None, Stroke = 4f };
                var image = implementation.Draw(new DocumentPage(0, Width, Height) { Paths = [line] }, options).Image;
                checks.Check("a stroke inks its line and nothing far from it", Grey(image, 60f, 45f) <= Ink && Grey(image, 60f, 55f) >= 1f - Ink && Grey(image, 60f, 35f) >= 1f - Ink,
                    $"on the line {Grey(image, 60f, 45f)}, 10 points off it {Grey(image, 60f, 55f)} and {Grey(image, 60f, 35f)}");
                break;
            }

            case "image":
            {
                var red = new ImageData(Fill(3, 4, 4, [1f, 0f, 0f]), 3, 4, 4);
                var page = new DocumentPage(0, Width, Height) { Images = [ImageOf(red, new DocumentBox(30f, 20f, 40f, 40f))] };
                var image = Same(implementation, page, options, checks).Image;
                checks.Check("an image fills its box with its colour", Close(image, 0, 100, 80, 1f) && Close(image, 1, 100, 80, 0f) && Close(image, 2, 100, 80, 0f)
                    && Close(image, 1, 10, 10, 1f), "the image's middle is not red, or the page outside it is not white");
                break;
            }

            case "glyphs" when Face is null:
                checks.Skip("a glyph run matches the rasterizer's coverage", "no font given");
                break;

            case "glyphs":
            {
                var face = Face!;
                var run = GlyphRun(face, "Hello", 30f, 10f, 50f, DocumentTextMode.Fill);
                var page = new DocumentPage(0, Width, Height) { Runs = [run] };
                var drawing = Same(implementation, page, options, checks);
                float scale = run.FontSize / face.Metrics.UnitsPerEm * options.PixelsPerPoint;
                CheckCoverage(drawing.Image, [.. run.Glyphs.ToArray().Select(g => (face.Outline(g.Glyph), scale, g.X * options.PixelsPerPoint, g.Y * options.PixelsPerPoint))],
                    options, checks, "a glyph run matches the rasterizer's coverage");
                break;
            }

            case "missing-font":
            {
                var run = new DocumentTextRun("Hello", new DocumentBox(10f, 30f, 60f, 20f), 20f)
                {
                    Font = "A font nobody has", Glyphs = new DocumentGlyph[] { new(1, 10f, 45f, 10f), new(2, 20f, 45f, 10f) }, Baseline = 45f,
                };
                try
                {
                    var drawing = implementation.Draw(new DocumentPage(0, Width, Height) { Runs = [run] }, options);
                    checks.Check("a missing font is counted, not thrown", drawing.Issues.Any(i => i.Kind == DocumentIssue.MissingFont && i.Count >= 1),
                        $"issues: {string.Join(", ", drawing.Issues.Select(i => $"{i.Kind} {i.Count}"))}");
                    checks.Check("its run is left out", drawing.Image.Pixels.All(v => v >= 1f - Ink), "something was drawn for a run with no font");
                }
                catch (Exception e)
                {
                    checks.Fail("a missing font is counted, not thrown", $"{e.GetType().Name}: {e.Message}");
                }

                break;
            }

            case "parts":
            {
                var square = DocumentPath.Rectangle(new DocumentBox(5f, 5f, 30f, 30f));
                var blue = ImageOf(new ImageData(Fill(3, 2, 2, [0f, 0f, 1f]), 3, 2, 2), new DocumentBox(45f, 5f, 30f, 30f));
                DocumentTextRun[] runs = Face is null ? [] : [GlyphRun(Face, "H", 30f, 85f, 35f, DocumentTextMode.Fill)];
                var page = new DocumentPage(0, Width, Height) { Paths = [square], Images = [blue], Runs = runs };
                var paths = implementation.Draw(page, options with { Parts = DocumentParts.Paths }).Image;
                var images = implementation.Draw(page, options with { Parts = DocumentParts.Images }).Image;
                checks.Check("paths alone when asked", Grey(paths, 20f, 20f) <= Ink && Grey(paths, 60f, 20f) >= 1f - Ink && Ink3(paths, 80f, 10f, 100f, 40f) == 0,
                    $"the square {Grey(paths, 20f, 20f)}, the image's place {Grey(paths, 60f, 20f)}");
                checks.Check("images alone when asked", Grey(images, 20f, 20f) >= 1f - Ink && Close(images, 2, 120, 40, 1f) && Close(images, 0, 120, 40, 0f) && Ink3(images, 80f, 10f, 100f, 40f) == 0,
                    $"the square's place {Grey(images, 20f, 20f)}, the image not blue");
                if (Face is null)
                {
                    checks.Skip("text alone when asked", "no font given");
                }
                else
                {
                    var text = implementation.Draw(page, options with { Parts = DocumentParts.Text }).Image;
                    checks.Check("text alone when asked", Grey(text, 20f, 20f) >= 1f - Ink && Grey(text, 60f, 20f) >= 1f - Ink && Ink3(text, 80f, 10f, 110f, 40f) > 0,
                        "the text page holds the square or the image, or no text");
                }

                break;
            }

            case "hidden" when Face is null:
                checks.Skip("hidden text drawn only when asked", "no font given");
                break;

            case "hidden":
            {
                var page = new DocumentPage(0, Width, Height) { Runs = [GlyphRun(Face!, "Hidden", 30f, 10f, 50f, DocumentTextMode.Hidden)] };
                var unasked = implementation.Draw(page, options).Image;
                var asked = implementation.Draw(page, options with { Hidden = true }).Image;
                checks.Check("hidden text drawn only when asked", unasked.Pixels.All(v => v >= 1f - Ink) && Ink3(asked, 0f, 0f, Width, Height) > 0,
                    "hidden text drawn unasked, or not drawn when asked");
                break;
            }

            case "grey":
            {
                var page = new DocumentPage(0, Width, Height) { Paths = [Rectangle(new DocumentBox(10f, 10f, 50f, 50f), DocumentColour.Grey(0.5f), 0)] };
                var image = implementation.Draw(page, options with { Colour = DocumentDrawColour.Grey }).Image;
                checks.Check("grey pixels when asked", image.Channels == 1 && MathF.Abs(DocumentChecks.Pixel(image, 0, 70, 70) - 0.5f) <= Ink * 2,
                    $"{image.Channels} channels, the square {DocumentChecks.Pixel(image, 0, 70, 70)}");
                break;
            }

            default:
                RunRectangles(implementation, new Random((int)@case.Data["seed"]!), options, checks);
                break;
        }
    }

    private static void RunRectangles(IDocumentPageDrawer implementation, Random random, DocumentDrawOptions options, CaseChecks checks)
    {
        var paths = new List<DocumentPath>();
        var colours = new List<(float X, float Y, DocumentColour Colour)>();
        for (int row = 0; row < 3; row++)
        {
            for (int column = 0; column < 4; column++)
            {
                var colour = DocumentColour.Rgb(random.Next(256) / 255f, random.Next(256) / 255f, random.Next(256) / 255f);
                var box = new DocumentBox(5f + column * 28f, 5f + row * 28f, 20f + random.Next(4), 20f + random.Next(4));
                paths.Add(Rectangle(box, colour, paths.Count));
                colours.Add((box.X + box.Width / 2, box.Y + box.Height / 2, colour));
            }
        }

        var image = implementation.Draw(new DocumentPage(0, Width, Height) { Paths = paths }, options).Image;
        int wrong = colours.FindIndex(c => !Close(image, 0, (int)(c.X * 2), (int)(c.Y * 2), c.Colour.Red) || !Close(image, 1, (int)(c.X * 2), (int)(c.Y * 2), c.Colour.Green)
                                           || !Close(image, 2, (int)(c.X * 2), (int)(c.Y * 2), c.Colour.Blue));
        checks.Check("each rectangle's middle has its colour", wrong < 0, wrong < 0 ? "" : $"the rectangle at ({colours[wrong].X}, {colours[wrong].Y}) is not {colours[wrong].Colour}");
    }

    // Draws the page twice and from two threads: the same pixels each time.
    private static DocumentDrawing Same(IDocumentPageDrawer implementation, DocumentPage page, DocumentDrawOptions options, CaseChecks checks)
    {
        var drawing = implementation.Draw(page, options);
        checks.Compare("the same drawing again", Comparisons.Difference(drawing.Image.Pixels, implementation.Draw(page, options).Image.Pixels, 0f));
        var parallel = new DocumentDrawing[2];
        Parallel.For(0, 2, i => parallel[i] = implementation.Draw(page, options));
        checks.Compare("the same drawing from several threads",
            Comparisons.Difference(drawing.Image.Pixels, parallel[0].Image.Pixels, 0f) ?? Comparisons.Difference(drawing.Image.Pixels, parallel[1].Image.Pixels, 0f));
        return drawing;
    }

    // The drawing's grey against 1 - the coverage the registered rasterizer gives the outlines (black on white).
    private void CheckCoverage(ImageData image, IReadOnlyList<(GlyphOutline Outline, float Scale, float X, float Y)> outlines, DocumentDrawOptions options, CaseChecks checks, string property)
    {
        if (OutlineRasterizers.Names.Count == 0 || !OutlineRasterizers.Names.Contains(options.Rasterizer, StringComparer.OrdinalIgnoreCase))
        {
            checks.Skip(property, $"no rasterizer '{options.Rasterizer}' is registered to compare with");
            return;
        }

        var coverage = new float[image.Width * image.Height];
        var rasterizer = OutlineRasterizers.Get(options.Rasterizer);
        foreach (var (outline, scale, x, y) in outlines)
        {
            rasterizer.Fill(outline, scale, x, y, coverage, image.Width, image.Height);
        }

        double total = 0;
        for (int y = 0; y < image.Height; y++)
        {
            for (int x = 0; x < image.Width; x++)
            {
                total += MathF.Abs(DocumentChecks.Grey(image, x, y) - (1f - coverage[y * image.Width + x]));
            }
        }

        double mean = total / coverage.Length;
        checks.Check(property, mean <= Tolerance, $"the mean difference from the rasterizer's coverage is {mean:G4}, more than {Tolerance}");
    }

    // A page path as an outline the rasterizer takes: y up, so with a scale of the pixels per point and the origin at the
    // page's top-left corner it lands where the page puts it.
    private static GlyphOutline Outline(DocumentPath path)
    {
        float[] points = path.Points.ToArray();
        for (int i = 1; i < points.Length; i += 2)
        {
            points[i] = -points[i];
        }

        return new GlyphOutline(path.Verbs.ToArray(), points);
    }

    private static DocumentTextRun GlyphRun(IFontFace face, string text, float size, float x, float baseline, DocumentTextMode mode)
    {
        var glyphs = new List<DocumentGlyph>();
        float scale = size / face.Metrics.UnitsPerEm, at = x;
        foreach (var rune in text.EnumerateRunes())
        {
            int glyph = face.Glyph(rune.Value);
            float advance = face.Advance(glyph) * scale;
            glyphs.Add(new DocumentGlyph(glyph, at, baseline, advance));
            at += advance;
        }

        return new DocumentTextRun(text, new DocumentBox(x, baseline - size, at - x, size * 1.25f), size)
        {
            Glyphs = glyphs.ToArray(), Face = face, Mode = mode, Baseline = baseline, Font = face.Family,
        };
    }

    private static DocumentPath Rectangle(DocumentBox box, DocumentColour colour, int order)
    {
        var rectangle = DocumentPath.Rectangle(box);
        return new DocumentPath(rectangle.Verbs.ToArray(), rectangle.Points.ToArray()) { FillColour = colour, Order = order };
    }

    private static DocumentImage ImageOf(ImageData pixels, DocumentBox box) =>
        new(DocumentImage.Raw, pixels.Width, pixels.Height, () => ReadOnlyMemory<byte>.Empty, () => pixels) { Box = box };

    private static float[] Fill(int channels, int height, int width, float[] colour)
    {
        var pixels = new float[channels * height * width];
        for (int c = 0; c < channels; c++)
        {
            pixels.AsSpan(c * height * width, height * width).Fill(colour[c]);
        }

        return pixels;
    }

    // The grey at a page point (pixels per point 2).
    private static float Grey(ImageData image, float x, float y) => DocumentChecks.Grey(image, (int)(x * 2f), (int)(y * 2f));

    private static bool Close(ImageData image, int channel, int x, int y, float expected) => MathF.Abs(DocumentChecks.Pixel(image, channel, x, y) - expected) <= 2 * Ink;

    // The pixels with ink (grey below white) in a box of page points.
    private static int Ink3(ImageData image, float left, float top, float right, float bottom)
    {
        int count = 0;
        for (int y = (int)(top * 2); y < Math.Min(image.Height, (int)(bottom * 2)); y++)
        {
            for (int x = (int)(left * 2); x < Math.Min(image.Width, (int)(right * 2)); x++)
            {
                count += DocumentChecks.Grey(image, x, y) < 1f - Ink ? 1 : 0;
            }
        }

        return count;
    }
}
