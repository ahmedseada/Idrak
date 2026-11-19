// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using Idrak.Imaging;

// Plan 28, D13's finding on the drawer's clips (clip memory, 2026-11-19): a band works a clip's coverage once, over the
// clip's own box (an upright rectangle as the rows that differ), and gives it back to the pool after the last part that
// needs it, so a page's clips cost their outlines, never a band of floats each. These tests hold the memory to a bound
// that does not grow with the clips, and the drawings to the band-wide masks' value for value.
internal abstract partial class Tests
{
    // What rendering every band of a page may allocate beyond what it rents from the shared pools, whatever its clips (each
    // band's own bookkeeping: its work, its dictionaries).
    private const long ClipRenderBytes = 64 * 1024;

    // What one clip adds to a draw: its outline made ready in the surface's pixels and its record; its coverage is rented.
    private const long ClipSceneBytes = 1024;

    // What drawing the large set's deep-forms page (one form drawn 1,000 times, each under its /BBox) may allocate at 150 dpi
    // beyond its pixels: its 1,000 parts and 1,000 clips made ready (before the clips' memory was cut: about 372 MiB).
    private const long DeepFormsDrawBytes = 8L << 20;

    private static void PdfDrawClipMemory(Device device)
    {
        CpuOnly(device);
        var options = new DocumentDrawOptions { Dpi = 150, Fonts = null };
        var failures = new List<string>();
        foreach (int wanted in (int[])[10, 100, 1000])
        {
            int parts = ClipMemoryParts(wanted);
            var clipped = ClipMemoryPage(parts, clips: true, options);
            var plain = ClipMemoryPage(parts, clips: false, options);
            int clips = clipped.Clips.Count;
            Check(clips >= wanted && clipped.Paths.Count == parts && plain.Paths.Count == parts && plain.Clips.Count == 0,
                $"the written page of {wanted} clips read back with {clips} clips and {clipped.Paths.Count} parts ({plain.Paths.Count} unclipped)");
            long render = ClipRenderAllocated(clipped, options);
            var (drawn, drawnMs) = ClipDrawAllocated(clipped, options);
            var (plainDrawn, plainMs) = ClipDrawAllocated(plain, options);
            long added = drawn - plainDrawn;
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"    {clips} clips over {parts} parts at 150 dpi: every band rendered {render / 1024.0:F1} KiB past the pools; a draw {drawn / 1048576.0:F2} MiB (unclipped {plainDrawn / 1048576.0:F2}), the clips {added / (double)clips:F0} B each; warm {drawnMs:F1} ms (unclipped {plainMs:F1})"));
            if (render > ClipRenderBytes)
            {
                failures.Add($"{clips} clips: rendering the bands allocated {render} bytes past the pools, more than {ClipRenderBytes} ({AllocatedTypes(ClipRenderBands(clipped, options), 1 << 20, 50)})");
            }

            if (added > clips * ClipSceneBytes + ClipRenderBytes)
            {
                failures.Add($"{clips} clips added {added} bytes to a draw, more than {ClipSceneBytes} a clip and {ClipRenderBytes}");
            }
        }

        // The large set's deep-forms page, when the set is there (made by make_fixtures.py --large).
        if (ClipDeepForms() is not { } deep)
        {
            Console.WriteLine("    the large set's deep-forms.pdf is not there (python -I tools/documents/make_fixtures.py -o test-output/documents/samples --large): its page left out");
        }
        else
        {
            using var document = DocumentFormats.Open(deep);
            var page = document.ReadPage(3, DocumentPageDrawers.PartsFor(options));
            Check(page.Clips.Count >= 1000, $"deep-forms page 4: {page.Clips.Count} clips, the generator draws its form 1,000 times");
            long start = Stopwatch.GetTimestamp();
            var cold = DocumentPageDrawers.Draw(page, options);
            double coldMs = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            long pixels = (long)cold.Image.Width * cold.Image.Height * cold.Image.Channels;
            var (drawn, warmMs) = ClipDrawAllocated(page, options);
            long render = ClipRenderAllocated(page, options);
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"    deep-forms page 4 ({page.Clips.Count} clips) at 150 dpi: cold {coldMs:F1} ms, warm {warmMs:F1} ms and {drawn / 1048576.0:F2} MiB a draw ({pixels / 1048576.0:F2} MiB its pixels); every band rendered {render / 1024.0:F1} KiB past the pools"));
            if (drawn - pixels > DeepFormsDrawBytes || render > ClipRenderBytes)
            {
                failures.Add($"deep-forms page 4: a draw allocated {drawn - pixels} bytes beyond its pixels (at most {DeepFormsDrawBytes}), its bands {render} (at most {ClipRenderBytes})");
            }
        }

        Check(failures.Count == 0, $"{failures.Count} clip memory bounds exceeded:\n      {string.Join("\n      ", failures)}");
    }

    // The fewest parts whose clips reach `wanted`, one page clip besides: each part's clips in turn one, one, three.
    private static int ClipMemoryParts(int wanted)
    {
        int parts = 0;
        for (int clips = 1; clips < wanted; parts++)
        {
            clips += parts % 3 == 2 ? 3 : 1;
        }

        return parts;
    }

    // A page of 420 x 300 points as the pdf writer writes it and the reader reads it back: `parts` squares of 6 points on a
    // 10-point grid, with `clips` each under clips of its own within one page clip, in turn an upright rectangle around its
    // cell, a turned one, or three nested (a rectangle, a turned one inside it, a rectangle inside that); without, unclipped.
    private static DocumentPage ClipMemoryPage(int parts, bool clips, DocumentDrawOptions options)
    {
        var colour = DocumentColour.Rgb(0.2f, 0.4f, 0.8f);
        byte[] written = PdfWriteDocument(b =>
        {
            var page = b.AddPage(420f, 300f);
            if (clips)
            {
                page.Clip(DocumentPath.Rectangle(new DocumentBox(0.3f, 0.3f, 419.4f, 299.4f)));
            }

            for (int k = 0; k < parts; k++)
            {
                float x = 10f + k % 40 * 10f, y = 10f + k / 40 * 10f;
                if (clips)
                {
                    page.Save();
                    var cell = new DocumentBox(x - 1.37f, y - 1.21f, 8.6f, 8.45f);
                    switch (k % 3)
                    {
                        case 0:
                            page.Clip(DocumentPath.Rectangle(cell));
                            break;
                        case 1:
                            page.Clip(ClipTurned(cell, 4f));
                            break;
                        default:
                            page.Clip(DocumentPath.Rectangle(cell));
                            page.Clip(ClipTurned(new DocumentBox(x - 0.8f, y - 0.6f, 7.4f, 7.3f), 3f));
                            page.Clip(DocumentPath.Rectangle(new DocumentBox(x + 0.45f, y + 0.35f, 5.2f, 5.3f)));
                            break;
                    }
                }

                page.Path(new DocumentPath(DocumentPath.Rectangle(new DocumentBox(x, y, 6f, 6f)).Verbs.ToArray(), DocumentPath.Rectangle(new DocumentBox(x, y, 6f, 6f)).Points.ToArray())
                {
                    FillColour = colour,
                });
                if (clips)
                {
                    page.Restore();
                }
            }
        });
        using var document = DocumentFormats.Open(written.AsMemory(), "clip-memory.pdf");
        return document.ReadPage(0, DocumentPageDrawers.PartsFor(options));
    }

    // A rectangle turned about its centre by `degrees`, as a closed path of four lines.
    private static DocumentPath ClipTurned(DocumentBox box, float degrees)
    {
        float cx = box.X + box.Width / 2, cy = box.Y + box.Height / 2, c = MathF.Cos(degrees * MathF.PI / 180f), s = MathF.Sin(degrees * MathF.PI / 180f);
        float[] corners = [box.X, box.Y, box.Right, box.Y, box.Right, box.Bottom, box.X, box.Bottom];
        for (int i = 0; i < corners.Length; i += 2)
        {
            float dx = corners[i] - cx, dy = corners[i + 1] - cy;
            (corners[i], corners[i + 1]) = (cx + c * dx - s * dy, cy + s * dx + c * dy);
        }

        return new DocumentPath([OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close], corners);
    }

    // The scene of `page` rendered band by band on this thread, warm (the pools hold what a band rents): the bytes allocated.
    private static long ClipRenderAllocated(DocumentPage page, DocumentDrawOptions options)
    {
        var bands = ClipRenderBands(page, options);
        for (int warm = 0; warm < 3; warm++)
        {
            bands();
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        bands();
        return GC.GetAllocatedBytesForCurrentThread() - before;
    }

    // Renders every band of `page`'s scene, the drawer's bands, one after another into one surface.
    private static Action ClipRenderBands(DocumentPage page, DocumentDrawOptions options)
    {
        var (width, height) = DocumentPageDrawers.PixelSize(page, options);
        var scene = DocumentScene.Build(page, new Affine(options.PixelsPerPoint, 0, 0, options.PixelsPerPoint, 0, 0), width, height, new DrawContext(options), options.Parts, hidden: false);
        int rows = CoverageDocumentDrawer.BandRows(width, options.BandPixels, options.MinBandRows, options.MaxBandRows);
        var surface = new float[4 * width * rows];
        return () =>
        {
            for (int top = 0; top < height; top += rows)
            {
                int count = Math.Min(rows, height - top);
                var band = surface.AsSpan(0, 4 * width * count);
                band.Clear();
                scene.Render(band, top, count);
            }
        };
    }

    // A warm draw of `page` through the drawers' table: the bytes allocated on every thread and the median time of three.
    private static (long Bytes, double Milliseconds) ClipDrawAllocated(DocumentPage page, DocumentDrawOptions options)
    {
        GC.KeepAlive(DocumentPageDrawers.Draw(page, options));
        var times = new List<double>();
        long bytes = long.MaxValue;
        for (int run = 0; run < 3; run++)
        {
            long before = GC.GetTotalAllocatedBytes(precise: true);
            long start = Stopwatch.GetTimestamp();
            var drawing = DocumentPageDrawers.Draw(page, options);
            times.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds);
            bytes = Math.Min(bytes, GC.GetTotalAllocatedBytes(precise: true) - before);
            GC.KeepAlive(drawing);
        }

        times.Sort();
        return (bytes, times[1]);
    }

    // The large set's deep-forms.pdf (beside the samples' folder, as make_fixtures.py --large writes it), or null.
    private static string? ClipDeepForms()
    {
        string? samples = Environment.GetEnvironmentVariable("IDRAK_DOCUMENTS_SAMPLES") is { Length: > 0 } set ? set
            : TestPaths.Root is { } root ? Path.Combine(root, "test-output", "documents", "samples") : null;
        if (samples is null)
        {
            return null;
        }

        string path = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(samples).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar))!, "large", "d13", "deep-forms.pdf");
        return File.Exists(path) ? path : null;
    }

    // ---- clips draw as they did ----------------------------------------------------------------------------------------

    private static void PdfDrawClipsExact(Device device)
    {
        CpuOnly(device);
        PdfDrawClipValues();
        PdfDrawClipPages();
    }

    // Each clip, alone and in chains, filled over its own box by the library's rasterizer (ClipCover.Bounded) and over the
    // whole band (ClipCover.Banded, the drawer's way before and still for any other rasterizer): every value the same, in
    // bands of 1, 7, 64 and all 83 rows.
    private static void PdfDrawClipValues()
    {
        const int width = 97, height = 83;
        var rasterizer = new CoverageRasterizer();
        static GlyphOutline Box(float x0, float y0, float x1, float y1) =>
            new([OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close], [x0, -y0, x1, -y0, x1, -y1, x0, -y1]);
        static GlyphOutline Turned(float cx, float cy, float w, float h, float degrees)
        {
            var path = ClipTurned(new DocumentBox(cx - w / 2, cy - h / 2, w, h), degrees);
            float[] points = path.Points.ToArray();
            for (int i = 1; i < points.Length; i += 2)
            {
                points[i] = -points[i];
            }

            return new GlyphOutline(path.Verbs.ToArray(), points);
        }

        var shapes = new (string Name, GlyphOutline Outline, OutlineFill Rule, bool Upright)[]
        {
            ("an upright rectangle at sub-pixel edges", Box(10.13f, 7.37f, 80.6f, 57.71f), OutlineFill.NonZero, true),
            ("an upright rectangle at whole pixels", Box(12f, 20f, 70f, 75f), OutlineFill.NonZero, true),
            ("a rectangle smaller than a pixel", Box(40.2f, 30.1f, 40.5f, 30.3f), OutlineFill.NonZero, true),
            ("a rectangle of no width", Box(30f, 10f, 30f, 60f), OutlineFill.NonZero, true),
            ("a rectangle of no height", Box(20f, 40f, 80f, 40f), OutlineFill.NonZero, true),
            ("a rectangle off the surface", Box(-50f, -40f, -10f, -5f), OutlineFill.NonZero, true),
            ("a rectangle past every edge", Box(-10.5f, -3.25f, 120.75f, 99.5f), OutlineFill.NonZero, true),
            ("a rectangle wound the other way, closed by a fifth line", new GlyphOutline(
                [OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close],
                [15.6f, -12.4f, 15.6f, -66.9f, 75.3f, -66.9f, 75.3f, -12.4f, 15.6f, -12.4f]), OutlineFill.NonZero, true),
            ("an upright rectangle by the even-odd rule", Box(20.5f, 15.5f, 70.25f, 60.75f), OutlineFill.EvenOdd, true),
            ("a rectangle turned 3 degrees", Turned(48f, 41f, 60f, 40f, 3f), OutlineFill.NonZero, false),
            ("a rectangle turned 45 degrees", Turned(45f, 40f, 40f, 40f, 45f), OutlineFill.NonZero, false),
            ("a circle", DrawSettingsCircle(50f, 40f, 25.3f), OutlineFill.NonZero, false),
            ("an even-odd frame", new GlyphOutline(
                [OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close, OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close],
                [8f, -6f, 88f, -6f, 88f, -76f, 8f, -76f, 30f, -25f, 66f, -25f, 66f, -55f, 30f, -55f]), OutlineFill.EvenOdd, false),
        };
        var chains = new List<int[]>();
        chains.AddRange(Enumerable.Range(0, shapes.Length).Select(i => new[] { i }));
        chains.AddRange(
        [
            [0, 1], [1, 0], [6, 0], [0, 9], [10, 0], [11, 1], [12, 6], [0, 2], [1, 8], [7, 12], [0, 5], [3, 0],
            [6, 11, 0, 9, 1, 8],
        ]);

        var failures = new List<string>();
        int shared = 0, compared = 0;
        foreach (var chain in chains)
        {
            string name = string.Join(" within ", chain.Select(i => shapes[i].Name).Reverse());
            var clips = new SceneClip[chain.Length];
            for (int i = 0; i < chain.Length; i++)
            {
                var shape = shapes[chain[i]];
                clips[i] = new SceneClip(shape.Outline, shape.Rule, i - 1, false, ClipBox.Of(shape.Outline, width, height));
                if (!clips[i].Box.IsEmpty && clips[i].Box.Rectangle != shape.Upright)
                {
                    failures.Add($"{shape.Name}: read as {(clips[i].Box.Rectangle ? "an upright rectangle" : "a general clip")}");
                }
            }

            foreach (int rows in (int[])[1, 7, 64, height])
            {
                for (int top = 0; top < height; top += rows)
                {
                    int count = Math.Min(rows, height - top);
                    var work = new SceneWork(width, top, count, [], [], rasterizer, 1f / 32);
                    var banded = new ClipCover[clips.Length];
                    var bounded = new ClipCover[clips.Length];
                    try
                    {
                        for (int i = 0; i < clips.Length; i++)
                        {
                            banded[i] = ClipCover.Banded(clips[i], i > 0 ? banded[i - 1] : default, work);
                            bounded[i] = ClipCover.Bounded(clips[i], i > 0 ? bounded[i - 1] : default, work);
                            shared += bounded[i].Rows is not null ? 1 : 0;
                            string? differs = ClipDiffers(banded[i], bounded[i], width, top, count);
                            compared++;
                            if (differs is not null && failures.Count < 20)
                            {
                                failures.Add($"{name}, clip {i + 1} of {clips.Length}, a band of {count} rows at row {top}: {differs}");
                            }
                        }
                    }
                    finally
                    {
                        foreach (var cover in banded)
                        {
                            cover.Return();
                        }

                        foreach (var cover in bounded)
                        {
                            cover.Return();
                        }
                    }
                }
            }
        }

        Console.WriteLine($"    {chains.Count} clips and chains in bands of 1, 7, 64 and {height} rows: {compared} coverages compared value for value; {shared} kept a rectangle's rows once");
        Check(failures.Count == 0 && shared > 0, $"a clip's coverage over its own box is not the band-wide coverage ({failures.Count} differences; rectangles' rows shared {shared} times):\n      {string.Join("\n      ", failures)}");
    }

    // The first pixel where the bounded coverage (0 outside its box) differs from the band-wide one, or null.
    private static string? ClipDiffers(ClipCover banded, ClipCover bounded, int width, int top, int rows)
    {
        for (int y = top; y < top + rows; y++)
        {
            bool inRows = bounded.Clips && y >= bounded.Top && y < bounded.Bottom;
            ReadOnlySpan<float> row = inRows ? bounded.Row(y) : default;
            for (int x = 0; x < width; x++)
            {
                float wide = banded.Values![(y - top) * width + x];
                float own = inRows && x >= bounded.Left && x < bounded.Right ? row[x - bounded.Origin] : 0f;
                if (wide != own)
                {
                    return string.Create(CultureInfo.InvariantCulture, $"at ({x}, {y}) {own:R}, the band-wide fill {wide:R}");
                }
            }
        }

        return null;
    }

    // Crafted pages drawn through the library's rasterizer (each clip over its own box) and through an app's rasterizer
    // that forwards every fill to the library's (each clip over the whole band, as the drawer always did): the same bytes
    // and the same issues, at 150 dpi, at 2 pixels a point in grey, with one-row bands and with one band.
    private static void PdfDrawClipPages()
    {
        const string forward = "test-clips-band-wide";
        var page = ClipCraftedPage();
        var failures = new List<string>();
        OutlineRasterizers.Register(new ClipsForwardRasterizer(forward));
        try
        {
            foreach (var (name, options) in new (string, DocumentDrawOptions)[]
                     {
                         ("150 dpi", new DocumentDrawOptions { Dpi = 150, Fonts = null }),
                         ("2 pixels a point, grey", new DocumentDrawOptions { PixelsPerPoint = 2f, Colour = DocumentDrawColour.Grey, Fonts = null }),
                         ("150 dpi, one-row bands", new DocumentDrawOptions { Dpi = 150, Fonts = null, BandPixels = 1, MinBandRows = 1 }),
                         ("150 dpi, one band", new DocumentDrawOptions { Dpi = 150, Fonts = null, BandPixels = 1 << 26, MaxBandRows = 1 << 16 }),
                     })
            {
                var own = DocumentPageDrawers.Draw(page, options);
                var wide = DocumentPageDrawers.Draw(page, options with { Rasterizer = forward });
                byte[] a = PdfInterleaved(own.Image), b = PdfInterleaved(wide.Image);
                int differ = a.Length == b.Length ? a.Zip(b).Count(p => p.First != p.Second) : -1;
                string issues = string.Join("; ", own.Issues.Select(i => $"{i.Kind} {i.Detail} {i.Count}")), wideIssues = string.Join("; ", wide.Issues.Select(i => $"{i.Kind} {i.Detail} {i.Count}"));
                int inked = a.Count(v => v < 250);
                Console.WriteLine($"    the crafted page at {name}: {differ} of {a.Length} values differ from the band-wide clips' drawing; {inked} values inked");
                if (differ != 0 || issues != wideIssues || inked == 0 || inked == a.Length)
                {
                    failures.Add($"{name}: {differ} values differ, issues '{issues}' against '{wideIssues}', {inked} of {a.Length} inked");
                }
            }
        }
        finally
        {
            OutlineRasterizers.Unregister(forward);
        }

        Check(failures.Count == 0, $"the crafted page drawn with each clip over its own box is not the band-wide clips' drawing:\n      {string.Join("\n      ", failures)}");
    }

    // A page of 300 x 220 points with every kind of clip under parts that reach past it: upright rectangles at sub-pixel
    // and whole-pixel edges, smaller than a pixel, of no width or height, off the page and past its edges, closed by a fifth
    // line; turned ones, a circle, an even-odd frame; nested (each kind in another, six deep, one missing its parent); a
    // text clip whose runs are not there (clipping nothing of its own) with parts and a clip under it; a shading, a
    // shading-filled path, an image and a stroke larger than their clips; parts wholly outside their clips (one filled with a
    // tiling); clips shared by parts far apart in the order; a group and a soft mask with clips of its own.
    private static DocumentPage ClipCraftedPage()
    {
        var clips = new List<DocumentPath>();
        var paths = new List<DocumentPath>();
        int order = 0;
        int Clip(DocumentPath path, int parent = -1)
        {
            clips.Add(new DocumentPath(path.Verbs.ToArray(), path.Points.ToArray()) { Clip = parent, Fill = path.Fill, ClipRun = path.ClipRun, ClipRunCount = path.ClipRunCount });
            return clips.Count - 1;
        }

        static DocumentPath Rect(float x, float y, float w, float h, DocumentFill fill = DocumentFill.NonZero) =>
            new([OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close], [x, y, x + w, y, x + w, y + h, x, y + h]) { Fill = fill };
        void Fill(float x, float y, float w, float h, int clip, DocumentColour colour, int group = -1, int softMask = -1, DocumentPattern? pattern = null) =>
            paths.Add(new DocumentPath(Rect(x, y, w, h).Verbs.ToArray(), Rect(x, y, w, h).Points.ToArray())
            {
                FillColour = colour, Clip = clip, Order = order++, Group = group, SoftMask = softMask, FillPattern = pattern,
            });

        var red = DocumentColour.Rgb(0.85f, 0.15f, 0.1f);
        var green = DocumentColour.Rgb(0.1f, 0.6f, 0.2f);
        var blue = DocumentColour.Rgb(0.1f, 0.3f, 0.85f);
        var grey = DocumentColour.Grey(0.35f);

        // Upright rectangles.
        int subPixel = Clip(Rect(10.13f, 10.37f, 40.5f, 20.71f));
        Fill(5f, 5f, 55f, 35f, subPixel, red);
        int wholePixel = Clip(Rect(14.4f, 43.2f, 28.8f, 14.4f));
        Fill(5f, 40f, 50f, 25f, wholePixel, green);
        Fill(55f, 5f, 15f, 15f, Clip(Rect(60.2f, 10.1f, 0.3f, 0.2f)), blue);
        Fill(65f, 5f, 15f, 30f, Clip(Rect(70f, 10f, 0f, 20f)), red);
        Fill(75f, 5f, 30f, 15f, Clip(Rect(80f, 10f, 20f, 0f)), green);
        Fill(0f, 0f, 30f, 30f, Clip(Rect(-50f, -50f, 20f, 20f)), blue);
        Fill(250f, 5f, 50f, 30f, Clip(Rect(400f, 10f, 20f, 20f)), blue);
        Fill(60f, 40f, 40f, 25f, Clip(new DocumentPath(
            [OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close],
            [65.6f, 42.4f, 65.6f, 61.9f, 95.3f, 61.9f, 95.3f, 42.4f, 65.6f, 42.4f])), grey);

        // Turned rectangles, a circle, an even-odd frame.
        Fill(95f, 10f, 50f, 40f, Clip(ClipTurned(new DocumentBox(100f, 20f, 40f, 20f), 3f)), red);
        Fill(150f, 5f, 40f, 50f, Clip(ClipTurned(new DocumentBox(155f, 15f, 30f, 30f), 45f)), green);
        const float k = 0.5522848f;
        float cx = 220f, cy = 30f, r = 18f, c = k * r;
        Fill(195f, 5f, 50f, 50f, Clip(new DocumentPath(
            [OutlineVerb.Move, OutlineVerb.Cubic, OutlineVerb.Cubic, OutlineVerb.Cubic, OutlineVerb.Cubic, OutlineVerb.Close],
            [cx + r, cy, cx + r, cy + c, cx + c, cy + r, cx, cy + r, cx - c, cy + r, cx - r, cy + c, cx - r, cy, cx - r, cy - c, cx - c, cy - r, cx, cy - r, cx + c, cy - r, cx + r, cy - c, cx + r, cy])), blue);
        Fill(240f, 40f, 55f, 50f, Clip(new DocumentPath(
            [OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close, OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close],
            [245f, 45f, 290f, 45f, 290f, 85f, 245f, 85f, 255f, 55f, 280f, 55f, 280f, 75f, 255f, 75f]) { Fill = DocumentFill.EvenOdd }), red);

        // Nested: a rectangle in a rectangle, a turned one in a rectangle, a rectangle in a turned one, six deep, a miss.
        int outer = Clip(Rect(10.3f, 70.6f, 60.2f, 40.1f));
        Fill(0f, 60f, 80f, 70f, Clip(Rect(20.7f, 75.2f, 30.9f, 50f), outer), green);
        Fill(0f, 60f, 80f, 70f, Clip(ClipTurned(new DocumentBox(25f, 80f, 40f, 25f), 10f), outer), grey.WithAlpha(0.5f));
        int turned = Clip(ClipTurned(new DocumentBox(90f, 75f, 45f, 30f), 20f));
        Fill(85f, 65f, 60f, 50f, Clip(Rect(95.5f, 80.25f, 30f, 20f), turned), blue);
        int deep = Clip(Rect(150.25f, 65.5f, 80f, 60f));
        deep = Clip(ClipTurned(new DocumentBox(155f, 70f, 70f, 50f), 6f), deep);
        deep = Clip(Rect(158.6f, 72.3f, 60f, 45f), deep);
        deep = Clip(new DocumentPath([OutlineVerb.Move, OutlineVerb.Quadratic, OutlineVerb.Quadratic, OutlineVerb.Close], [160f, 95f, 190f, 50f, 220f, 95f, 190f, 140f, 160f, 95f]), deep);
        deep = Clip(Rect(165.1f, 80.9f, 50f, 30f), deep);
        deep = Clip(Rect(170.45f, 85.15f, 40.5f, 25.5f), deep);
        Fill(145f, 60f, 90f, 70f, deep, red);
        Fill(195f, 95f, 30f, 30f, Clip(Rect(250f, 100f, 20f, 20f), Clip(Rect(200f, 100f, 20f, 20f))), blue);

        // Past the page's edges, and across every band.
        Fill(0f, 0f, 300f, 220f, Clip(Rect(-10.5f, -10.25f, 14.3f, 300f)), green.WithAlpha(0.4f));
        Fill(270f, 120f, 30f, 100f, Clip(Rect(280.6f, 150.3f, 40f, 100f)), red);

        // Larger than their clips: a shading on its own, a shading-filled path, an image, a stroke.
        var shading = new DocumentShading(DocumentShadingKind.Axial, (t, _) => DocumentColour.Rgb(t, 0.5f, 1f - t))
        {
            Coords = new float[] { 0f, 0f, 300f, 220f }, ExtendStart = true, ExtendEnd = true, Order = order++, Clip = Clip(Rect(130.4f, 130.6f, 50.3f, 30.2f)),
        };
        Fill(185f, 125f, 50f, 45f, Clip(ClipTurned(new DocumentBox(190f, 130f, 40f, 30f), 15f)), red, pattern: new DocumentPattern(shading));
        float[] pixels = [.. Enumerable.Range(0, 3 * 8 * 6).Select(i => (i * 37 % 97) / 96f)];
        var image = new DocumentImage(DocumentImage.Raw, 8, 6, () => ReadOnlyMemory<byte>.Empty, () => new ImageData(pixels, 3, 6, 8))
        {
            Box = new DocumentBox(230f, 120f, 60f, 45f), Order = order++, Clip = Clip(Rect(240.25f, 125.5f, 30f, 20f)),
        };
        paths.Add(new DocumentPath([OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line], [10f, 185f, 70f, 215f, 90f, 175f])
        {
            Fill = DocumentFill.None, Stroke = 3f, StrokeColour = blue, Order = order++, Clip = Clip(Rect(10.5f, 180.5f, 60f, 25f)),
        });

        // Wholly outside their clips: a fill, and a fill by a tiling (its cell still drawn once, as before).
        Fill(150f, 190f, 20f, 20f, Clip(Rect(100f, 190f, 10f, 10f)), red);
        var cell = new DocumentPage(0, 10f, 10f) { Paths = [Rect(1f, 1f, 6f, 6f)] };
        Fill(175f, 190f, 30f, 20f, Clip(Rect(5f, 5f, 1f, 1f)), grey, pattern: new DocumentPattern(cell, new DocumentBox(0f, 0f, 10f, 10f), 10f, 10f, DocumentMatrix.Identity));

        // A text clip whose runs are not there, under a rectangle: the parts under it, and a clip made under it.
        int box = Clip(Rect(220.2f, 180.3f, 70f, 30f));
        int text = Clip(new DocumentPath([], []) { ClipRun = 9999, ClipRunCount = 1 }, box);
        Fill(215f, 175f, 40f, 20f, text, green);
        Fill(250f, 175f, 45f, 20f, text, blue);
        Fill(225f, 195f, 60f, 20f, Clip(Rect(230.5f, 200.5f, 50f, 15f), text), red);

        // Clips shared by parts far apart; a group and a soft mask with clips of their own.
        Fill(20f, 15f, 20f, 10f, subPixel, blue.WithAlpha(0.7f));
        Fill(20f, 45f, 20f, 10f, wholePixel, red.WithAlpha(0.7f));
        Fill(5f, 5f, 60f, 60f, subPixel, green.WithAlpha(0.5f), group: 0);
        Fill(90f, 5f, 60f, 60f, Clip(ClipTurned(new DocumentBox(100f, 20f, 40f, 20f), 3f)), red.WithAlpha(0.5f), group: 0);
        var mask = new DocumentSoftMask(new DocumentPage(0, 300f, 220f)
        {
            Paths = [new DocumentPath(Rect(100f, 140f, 80f, 60f).Verbs.ToArray(), Rect(100f, 140f, 80f, 60f).Points.ToArray()) { FillColour = DocumentColour.White, Clip = 0 }],
            Clips = [Rect(110.5f, 145.5f, 50f, 40f)],
        }, luminosity: true);
        Fill(90f, 135f, 100f, 70f, outer, blue, softMask: 0);
        Fill(90f, 135f, 100f, 70f, Clip(Rect(95.3f, 150.7f, 90f, 30f)), green, softMask: 0);

        return new DocumentPage(0, 300f, 220f)
        {
            Paths = paths, Clips = clips, Shadings = [shading], Images = [image],
            Groups = [new DocumentGroup { Isolated = true, Alpha = 0.6f }], SoftMasks = [mask],
        };
    }

    // An app's rasterizer that hands every fill to the library's: the drawer fills each clip over the whole band for it.
    private sealed class ClipsForwardRasterizer(string name) : IOutlineRasterizer
    {
        private readonly CoverageRasterizer _library = new();

        public string Name => name;

        public void Fill(GlyphOutline outline, float scale, float x, float y, Span<float> coverage, int width, int height) =>
            _library.Fill(outline, scale, x, y, coverage, width, height);

        public void Fill(GlyphOutline outline, float scale, float x, float y, Span<float> coverage, int width, int height, OutlineFill rule) =>
            _library.Fill(outline, scale, x, y, coverage, width, height, rule);

        public void Fill(GlyphOutline outline, float scale, float x, float y, Span<float> coverage, int width, int height, OutlineFill rule, float flatness) =>
            _library.Fill(outline, scale, x, y, coverage, width, height, rule, flatness);
    }
}
