// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Collections.Concurrent;
using System.Text;

namespace Idrak.Imaging;

/// <summary>
/// "coverage", the library's page drawer (<see cref="DocumentPageDrawers.Coverage"/>): any <see cref="DocumentPage"/>
/// drawn part by part in content order onto its pixels at <see cref="DocumentDrawOptions.PixelsPerPoint"/>. Fills by their
/// rule and strokes (the options' pen gives a stroke's points, <see cref="DocumentStroker"/> its outline: caps, joins,
/// miter limit, dashes) through the options' outline rasterizer (<see cref="OutlineRasterizers"/>, the nonzero and
/// even-odd rules); glyph runs through their faces' outlines (a run whose font the document does not embed from
/// <see cref="DocumentDrawOptions.Fonts"/>, by name and then by the characters it covers, else left out and counted);
/// images placed by their matrix (the homography of <see cref="PerspectiveWarp"/>: a magnified image's pixels as blocks, a
/// shrunk one's averaged), stencils in their colour, masks and opacity; shadings (axial, radial, function-based, meshes of
/// triangles) and tilings; clips (paths, and text); soft masks, transparency groups (isolated, knockout) and the sixteen
/// blend modes; then, when <see cref="DocumentParts.Annotations"/> is asked, each annotation's appearance (or, with none, its
/// look from its properties, <see cref="DocumentAnnotationLooks"/>) over the page's content, in the page's order. The page
/// is drawn band by band (rows in order), the bands spread over the cores: a large
/// page at a high resolution holds one band of coverage and colour a core, besides its pixels. Safe from several threads.
/// </summary>
public sealed class CoverageDocumentDrawer : IDocumentPageDrawer
{
    private static readonly float[] Levels = [.. Enumerable.Range(0, 256).Select(k => k * (1f / 255f))];

    /// <summary>"coverage".</summary>
    public string Name => DocumentPageDrawers.Coverage;

    /// <summary>One line on what it does.</summary>
    public string Summary => "a page's paths, glyph runs, images, shadings and tilings in content order, under their clips, soft masks and groups, by exact area coverage, band by band";

    /// <inheritdoc/>
    public DocumentDrawing Draw(DocumentPage page, DocumentDrawOptions options)
    {
        ArgumentNullException.ThrowIfNull(page);
        ArgumentNullException.ThrowIfNull(options);
        var (width, height) = DocumentPageDrawers.PixelSize(page, options);
        var context = new DrawContext(options);
        var map = new Affine(options.PixelsPerPoint, 0, 0, options.PixelsPerPoint, 0, 0);
        var scene = DocumentScene.Build(page, map, width, height, context, options.Parts, options.Hidden);

        // Annotations, when asked: each one's appearance (whole: it is the annotation's look), or its look from its
        // properties, over the page's content, in the page's order.
        var annotations = (options.Parts & DocumentParts.Annotations) != 0 ? Annotations(page, map, width, height, context, options.Hidden) : [];
        bool grey = options.Colour == DocumentDrawColour.Grey;
        var planes = new byte[grey ? 1 : 3][];
        for (int c = 0; c < planes.Length; c++)
        {
            planes[c] = GC.AllocateUninitializedArray<byte>(width * height);
        }

        var background = options.Background;
        int rows = BandRows(width, options.BandPixels, options.MinBandRows, options.MaxBandRows);
        int bands = (height + rows - 1) / rows;
        HostParallel.For(bands, 1, (first, last) =>
        {
            float[] surface = ArrayPool<float>.Shared.Rent(4 * width * rows);
            try
            {
                for (int band = first; band < last; band++)
                {
                    int top = band * rows, count = Math.Min(rows, height - top);
                    var target = surface.AsSpan(0, 4 * width * count);
                    Fill(target, background.Red, background.Green, background.Blue, 1f);
                    scene.Render(target, top, count);
                    foreach (var annotation in annotations)
                    {
                        annotation.Render(target, top, count);
                    }

                    Write(target, planes, top, count, width);
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(surface);
            }
        });

        return new DocumentDrawing(new ImageData(planes, height, width, Levels), context.Issues());
    }

    // The page's annotations as scenes drawn over its content, in the page's order: each by its appearance, or, with none,
    // by its look from its properties (DocumentAnnotationLooks); hidden ones, those not shown on screen and popups are not
    // drawn, and one drawn from its text alone (free text with no appearance) is counted as not drawn.
    private static List<DocumentScene> Annotations(DocumentPage page, Affine map, int width, int height, DrawContext context, bool hidden)
    {
        var scenes = new List<DocumentScene>();
        foreach (var annotation in page.Annotations)
        {
            if ((annotation.Flags & (DocumentAnnotationFlags.Hidden | DocumentAnnotationFlags.NoView)) != 0 || annotation.Kind == DocumentAnnotationKinds.Popup)
            {
                continue;
            }

            var look = annotation.Appearance ?? DocumentAnnotationLooks.Look(annotation, page.Width, page.Height);
            if (look is null)
            {
                if (annotation.Kind != DocumentAnnotationKinds.Widget)
                {
                    context.Issue(DocumentIssue.NotDrawn, $"page {page.Index + 1}: a {annotation.Kind} annotation with no appearance (its look comes from its text alone)");
                }

                continue;
            }

            scenes.Add(DocumentScene.Build(look, map, width, height, context, DocumentParts.Text | DocumentParts.Images | DocumentParts.Paths, hidden));
        }

        return scenes;
    }

    // A band holds about `pixels` pixels (DocumentDrawOptions.BandPixels, 128 Ki unless set: every buffer a band needs a
    // few MiB at most), from `minRows` to `maxRows` rows (8 to 1,024 unless set); refused when the bounds cross or a band's
    // colours would pass what one array holds.
    internal static int BandRows(int width, int pixels, int minRows, int maxRows)
    {
        if (minRows > maxRows)
        {
            throw new ArgumentOutOfRangeException(nameof(minRows), minRows, $"DocumentDrawOptions.MinBandRows ({minRows}) is more than MaxBandRows ({maxRows}): a band cannot hold both.");
        }

        int rows = Math.Clamp(pixels / Math.Max(1, width), minRows, maxRows);
        return 4L * Math.Max(1, width) * rows <= Array.MaxLength ? rows
            : throw new ArgumentOutOfRangeException(nameof(pixels), rows, $"A band of {rows} rows of {width} pixels is more than one array holds: lower DocumentDrawOptions.MinBandRows or BandPixels.");
    }

    internal static void Fill(Span<float> rgba, float r, float g, float b, float a)
    {
        for (int i = 0; i < rgba.Length; i += 4)
        {
            rgba[i] = r * a;
            rgba[i + 1] = g * a;
            rgba[i + 2] = b * a;
            rgba[i + 3] = a;
        }
    }

    // The band's colours as bytes (grey: the luma 0.299 R + 0.587 G + 0.114 B), rounded to the nearest level.
    private static void Write(ReadOnlySpan<float> rgba, byte[][] planes, int top, int rows, int width)
    {
        int at = top * width;
        int n = rows * width;
        if (planes.Length == 1)
        {
            var plane = planes[0].AsSpan(at, n);
            for (int i = 0; i < n; i++)
            {
                float v = 0.299f * rgba[4 * i] + 0.587f * rgba[4 * i + 1] + 0.114f * rgba[4 * i + 2];
                plane[i] = Byte(v);
            }

            return;
        }

        var red = planes[0].AsSpan(at, n);
        var green = planes[1].AsSpan(at, n);
        var blue = planes[2].AsSpan(at, n);
        for (int i = 0; i < n; i++)
        {
            red[i] = Byte(rgba[4 * i]);
            green[i] = Byte(rgba[4 * i + 1]);
            blue[i] = Byte(rgba[4 * i + 2]);
        }
    }

    private static byte Byte(float v) => (byte)(Math.Clamp(v, 0f, 1f) * 255f + 0.5f);
}

/// <summary>An affine map (x, y) to (A x + C y + E, B x + D y + F), in doubles.</summary>
internal readonly record struct Affine(double A, double B, double C, double D, double E, double F)
{
    public static Affine Identity { get; } = new(1, 0, 0, 1, 0, 0);

    public static Affine Of(DocumentMatrix m) => new(m.A, m.B, m.C, m.D, m.E, m.F);

    public (double X, double Y) Apply(double x, double y) => (A * x + C * y + E, B * x + D * y + F);

    /// <summary>This map, then <paramref name="next"/>.</summary>
    public Affine Then(Affine next) => new(next.A * A + next.C * B, next.B * A + next.D * B, next.A * C + next.C * D, next.B * C + next.D * D,
        next.A * E + next.C * F + next.E, next.B * E + next.D * F + next.F);

    public double Determinant => A * D - B * C;

    /// <summary>How far it stretches lengths on average: the square root of its determinant's size.</summary>
    public double Scale => Math.Sqrt(Math.Abs(Determinant));

    public bool Invertible => Determinant != 0 && double.IsFinite(Determinant);

    public Affine Inverse()
    {
        double det = Determinant;
        return new Affine(D / det, -B / det, -C / det, A / det, (C * F - D * E) / det, (B * E - A * F) / det);
    }
}

/// <summary>What one drawing uses: the options, the rasterizer, the pen and its settings, the folder's fonts, and the issues met (counted, from any band).</summary>
internal sealed class DrawContext(DocumentDrawOptions options)
{
    private readonly Lock _gate = new();
    private readonly Dictionary<(string Kind, string Detail), int> _issues = [];
    private readonly List<(string Kind, string Detail)> _order = [];
    private readonly Dictionary<string, IFontFace?> _substitutes = new(StringComparer.Ordinal);

    public DocumentDrawOptions Options { get; } = options;

    // The rasterizers and pens were registered (OutlineRasterizers, Pens) and picked by name: not on the new design yet.
    // The defaults, the "coverage" rasterizer and the "ideal" pen, are reached directly; another name stops.
    private static readonly CoverageRasterizer Coverage = new();

    private static readonly IdealPen Ideal = new();

    private static IOutlineRasterizer RasterizerNamed(string name) =>
        name.Equals(OutlineRasterizers.Coverage, StringComparison.OrdinalIgnoreCase) ? Coverage
        : throw new NotSupportedException($"The outline rasterizer '{name}' by name is not on the new design yet.");

    private static IPen PenNamed(string name) =>
        name.Equals(Pens.Ideal, StringComparison.OrdinalIgnoreCase) ? Ideal : throw new NotSupportedException($"The pen '{name}' by name is not on the new design yet.");

    public IOutlineRasterizer Rasterizer { get; } = RasterizerNamed(options.Rasterizer);

    public IPen Pen { get; } = PenNamed(options.Pen);

    public PenSettings PenSettings => _penSettings ??= Pen.Read(PageOptions.None);

    private PenSettings? _penSettings;

    public void Issue(string kind, string detail, int count = 1)
    {
        lock (_gate)
        {
            var key = (kind, detail);
            if (_issues.TryGetValue(key, out int seen))
            {
                _issues[key] = seen + count;
                return;
            }

            _issues[key] = count;
            _order.Add(key);
        }
    }

    public IReadOnlyList<DocumentIssue> Issues()
    {
        lock (_gate)
        {
            return [.. _order.Select(k => new DocumentIssue(k.Kind, k.Detail, _issues[k]))];
        }
    }

    /// <summary>A face from the options' folder for a run whose font is not embedded: by its name, then by the characters it covers; null when none.</summary>
    public IFontFace? Substitute(DocumentTextRun run)
    {
        if (Options.Fonts is not { Length: > 0 } folder)
        {
            return null;
        }

        string key = (run.Font ?? "") + "\u0001" + run.Text;
        lock (_gate)
        {
            if (_substitutes.TryGetValue(key, out var known))
            {
                return known;
            }
        }

        var face = FontFolder.Match(folder, run.Font, run.Text);
        lock (_gate)
        {
            _substitutes[key] = face;
        }

        return face;
    }
}

/// <summary>The faces of a folder of fonts, read once a process (rule 75), and a run's match among them.</summary>
internal static class FontFolder
{
    private static readonly ConcurrentDictionary<string, Lazy<IReadOnlyList<IFontFace>>> Folders = new(StringComparer.Ordinal);

    // A fonts folder was read through the registered font formats (FontFormats): not on the new design yet.
    private static IReadOnlyList<IFontFace> FontsIn(string folder) =>
        throw new NotSupportedException($"{folder}: reading fonts through the registered font formats is not on the new design yet.");

    public static IFontFace? Match(string folder, string? font, string text)
    {
        IReadOnlyList<IFontFace> faces;
        try
        {
            faces = Folders.GetOrAdd(Path.GetFullPath(folder), f => new Lazy<IReadOnlyList<IFontFace>>(() => Directory.Exists(f) ? FontsIn(f) : [])).Value;
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or InvalidDataException or NotSupportedException or ArgumentException)
        {
            return null;
        }

        if (faces.Count == 0)
        {
            return null;
        }

        // By name: the font's name without a subset's tag, letters and digits only, against each face's family and style.
        if (font is { Length: > 0 })
        {
            string name = Key(font.Length > 7 && font[6] == '+' ? font[7..] : font);
            foreach (var face in faces)
            {
                if (Key(face.Family + face.Style) == name || Key(face.Family) == name)
                {
                    return face;
                }
            }
        }

        // By coverage: the face that has the most of the run's characters (the first, in path order, of those that tie).
        IFontFace? best = null;
        int bestCount = -1;
        foreach (var face in faces)
        {
            int count = 0;
            foreach (var rune in text.EnumerateRunes())
            {
                count += Rune.IsWhiteSpace(rune) || face.Glyph(rune.Value) > 0 ? 1 : 0;
            }

            if (count > bestCount)
            {
                (best, bestCount) = (face, count);
            }
        }

        return bestCount > 0 ? best : null;
    }

    private static string Key(string name)
    {
        Span<char> buffer = stackalloc char[Math.Min(name.Length, 256)];
        int n = 0;
        foreach (char c in name.AsSpan(0, Math.Min(name.Length, 256)))
        {
            if (char.IsLetterOrDigit(c))
            {
                buffer[n++] = char.ToLowerInvariant(c);
            }
        }

        return new string(buffer[..n]);
    }
}
