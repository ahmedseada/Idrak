// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Data;

/// <summary>The pixels a page is drawn into.</summary>
public enum DocumentDrawColour : byte
{
    /// <summary>One grey channel.</summary>
    Grey,

    /// <summary>Red, green and blue.</summary>
    Rgb,
}

/// <summary>
/// How a page is drawn to pixels: the resolution, the colour, the parts drawn and the pieces the drawer draws with (all by
/// name, so each can be replaced). <see cref="DpiVariable"/> sets the resolution and <see cref="FontsVariable"/> the folder of
/// fonts for text whose font a document does not embed.
/// </summary>
public sealed record DocumentDrawOptions
{
    /// <summary>The variable that sets the resolution in dots per inch (<see cref="Dpi"/>; 150 unless set): "IDRAK_DOCUMENT_DPI".</summary>
    public const string DpiVariable = "IDRAK_DOCUMENT_DPI";

    /// <summary>The variable that names the folder of fonts (<see cref="Fonts"/>): "IDRAK_DOCUMENT_FONTS".</summary>
    public const string FontsVariable = "IDRAK_DOCUMENT_FONTS";

    private readonly float _pixelsPerPoint = (DocumentSettings.Float(DpiVariable, 1f, 9600f) ?? 150f) / 72f;

    /// <summary>Pixels per point (1/72 inch): <see cref="Dpi"/> / 72.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Not positive and finite.</exception>
    public float PixelsPerPoint
    {
        get => _pixelsPerPoint;
        init => _pixelsPerPoint = value > 0f && float.IsFinite(value) ? value : throw new ArgumentOutOfRangeException(nameof(PixelsPerPoint), value, "Pixels per point are more than 0.");
    }

    /// <summary>The resolution in dots per inch (150 unless <see cref="DpiVariable"/> sets it): <see cref="PixelsPerPoint"/> × 72.</summary>
    public float Dpi
    {
        get => _pixelsPerPoint * 72f;
        init => PixelsPerPoint = value / 72f;
    }

    /// <summary>Grey or RGB pixels (<see cref="DocumentDrawColour.Rgb"/>).</summary>
    public DocumentDrawColour Colour { get; init; } = DocumentDrawColour.Rgb;

    /// <summary>The colour under the page's content (white).</summary>
    public DocumentColour Background { get; init; } = DocumentColour.White;

    /// <summary>
    /// A folder of fonts for runs whose font the document does not embed, matched by name and then by the characters they
    /// cover (<see cref="FontsVariable"/>; null: such runs are left out and counted in the drawing's issues).
    /// </summary>
    public string? Fonts { get; init; } = DocumentSettings.Text(FontsVariable);

    /// <summary>The parts drawn: any of text, images, paths and annotations (text, images and paths).</summary>
    public DocumentParts Parts { get; init; } = DocumentParts.Text | DocumentParts.Images | DocumentParts.Paths;

    /// <summary>Whether invisible text (a scan's text layer) is drawn as if it were painted (false).</summary>
    public bool Hidden { get; init; }

    /// <summary>The drawer that draws the page: a <see cref="DocumentPageDrawers"/> name (<see cref="DocumentPageDrawers.Coverage"/>).</summary>
    public string Drawer { get; init; } = DocumentPageDrawers.Coverage;

    /// <summary>The rasterizer outlines are filled with: an <see cref="OutlineRasterizers"/> name (<see cref="OutlineRasterizers.Coverage"/>).</summary>
    public string Rasterizer { get; init; } = OutlineRasterizers.Coverage;

    /// <summary>The pen strokes are turned into outlines with: a <see cref="Pens"/> name (<see cref="Pens.Ideal"/>).</summary>
    public string Pen { get; init; } = Pens.Ideal;
}

/// <summary>A page drawn to pixels, and what could not be drawn (counted: a font that is missing, a part the drawer does not draw).</summary>
/// <param name="Image">The pixels.</param>
/// <param name="Issues">What was left out, and why.</param>
public sealed record DocumentDrawing(ImageData Image, IReadOnlyList<DocumentIssue> Issues);

/// <summary>
/// Draws any <see cref="DocumentPage"/> (its paths, glyph runs and images, in content order, under their clips) to pixels,
/// whatever format read it; register one with <see cref="DocumentPageDrawers.Register"/>. Safe to use from several threads
/// at once.
/// </summary>
public interface IDocumentPageDrawer : IKind
{
    /// <summary>
    /// <paramref name="page"/> drawn to an image of its size times <see cref="DocumentDrawOptions.PixelsPerPoint"/>
    /// (<see cref="DocumentPageDrawers.PixelSize"/>), on the options' background, with the options' parts. A run whose font
    /// is not there is left out and counted in the issues, never thrown.
    /// </summary>
    DocumentDrawing Draw(DocumentPage page, DocumentDrawOptions options);
}

/// <summary>
/// The page drawers, by name (ignoring case). Built in (Idrak.Imaging): <see cref="Coverage"/>. Register another with
/// <see cref="Register"/>; one registered under a library name takes its place, which stays behind it (see
/// <see cref="SetPolicy"/>).
/// </summary>
public static class DocumentPageDrawers
{
    /// <summary>"coverage": the library's drawer's name (exact area coverage through <see cref="OutlineRasterizers"/>).</summary>
    public const string Coverage = "coverage";

    /// <summary>How far two drawings' pixels may differ and still agree, under <see cref="SlotPolicy.Shadow"/>: two levels of 255.</summary>
    public const float ShadowTolerance = 2f / 255f;

    private static readonly KindTable<IDocumentPageDrawer> Table = new(nameof(DocumentPageDrawers), "Document page drawer",
        "add it with DocumentPageDrawers.Register, or reference a package that registers one (DocumentDrawOptions.Drawer names it)",
        (slot, app, library) => new GuardedDrawer(slot, app, library), StringComparer.OrdinalIgnoreCase);

    static DocumentPageDrawers() => LibraryDefaults.Ensure(typeof(DocumentPageDrawers));

    /// <summary>Registers <paramref name="drawer"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IDocumentPageDrawer drawer) => Table.Register(drawer, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's drawer <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Names;

    /// <summary>The drawer registered as <paramref name="name"/>, or null.</summary>
    public static IDocumentPageDrawer? Find(ReadOnlySpan<char> name) => Table.Find(name);

    /// <summary>The drawer registered as <paramref name="name"/>; an app's, guarded by its <see cref="SetPolicy">policy</see>.</summary>
    /// <exception cref="NotRegisteredException">None is: the message names the registered ones and what to do.</exception>
    public static IDocumentPageDrawer Get(ReadOnlySpan<char> name) => Table.Get(name);

    /// <summary>The library's drawer <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IDocumentPageDrawer? Default(string name) => Table.Default(name);

    /// <summary>Who registered the drawer <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>
    /// What happens when the app's drawer <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set;
    /// <see cref="SlotPolicy.FallBack"/> draws with the library's), or whether it only runs beside the library's
    /// (<see cref="SlotPolicy.Shadow"/>: the library's pixels are used, the two compared within <see cref="ShadowTolerance"/>).
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);

    /// <summary>Each registered drawer as "name: summary", one per line.</summary>
    public static string Describe() => Table.Describe();

    /// <summary>The image size a page is drawn at: its width and height times the pixels per point, rounded, at least one pixel each.</summary>
    public static (int Width, int Height) PixelSize(DocumentPage page, DocumentDrawOptions options)
    {
        ArgumentNullException.ThrowIfNull(page);
        ArgumentNullException.ThrowIfNull(options);
        return (Math.Max(1, (int)MathF.Round(page.Width * options.PixelsPerPoint)), Math.Max(1, (int)MathF.Round(page.Height * options.PixelsPerPoint)));
    }

    /// <summary>The parts a page is read with to be drawn: the options' parts, and the glyphs when text is drawn.</summary>
    public static DocumentParts PartsFor(DocumentDrawOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        return (options.Parts & DocumentParts.Text) != 0 ? options.Parts | DocumentParts.Glyphs : options.Parts;
    }

    /// <summary><paramref name="page"/> drawn by the drawer <paramref name="options"/> names.</summary>
    /// <exception cref="NotRegisteredException">That drawer is not registered.</exception>
    public static DocumentDrawing Draw(DocumentPage page, DocumentDrawOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(page);
        options ??= new DocumentDrawOptions();
        return Get(options.Drawer).Draw(page, options);
    }

    /// <summary>
    /// Page <paramref name="index"/> of <paramref name="document"/> read with the parts drawing needs and drawn: what
    /// <see cref="IDocument.DrawPage"/> does for a format whose pages can be drawn.
    /// </summary>
    /// <exception cref="NotRegisteredException">The drawer the options name is not registered.</exception>
    public static DocumentDrawing DrawPage(IDocument document, int index, DocumentDrawOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(document);
        options ??= new DocumentDrawOptions();
        var drawer = Get(options.Drawer);
        return drawer.Draw(document.ReadPage(index, PartsFor(options)), options);
    }

    /// <summary>
    /// Pages <paramref name="indices"/> of <paramref name="document"/> read with the parts drawing needs and drawn, in the
    /// order given: several at once over the cores (<see cref="ComputeResources.ParallelOptions"/>) when
    /// <paramref name="parallel"/>, else one after another (each page's own drawing still spread over the cores).
    /// </summary>
    /// <exception cref="NotRegisteredException">The drawer the options name is not registered.</exception>
    public static IReadOnlyList<DocumentDrawing> DrawPages(IDocument document, IReadOnlyList<int> indices, DocumentDrawOptions? options = null, bool parallel = true)
    {
        ArgumentNullException.ThrowIfNull(document);
        ArgumentNullException.ThrowIfNull(indices);
        options ??= new DocumentDrawOptions();
        var drawer = Get(options.Drawer);
        var parts = PartsFor(options);
        var drawings = new DocumentDrawing[indices.Count];
        if (!parallel || indices.Count < 2)
        {
            for (int i = 0; i < drawings.Length; i++)
            {
                drawings[i] = drawer.Draw(document.ReadPage(indices[i], parts), options);
            }

            return drawings;
        }

        Parallel.For(0, drawings.Length, ComputeResources.ParallelOptions, i => drawings[i] = drawer.Draw(document.ReadPage(indices[i], parts), options));
        return drawings;
    }

    // The library's drawing against the app's: the same size, every pixel within the tolerance, the same issue kinds.
    internal static string? CompareDrawings(DocumentDrawing expected, DocumentDrawing actual) =>
        Comparisons.Exact((expected.Image.Channels, expected.Image.Height, expected.Image.Width), (actual.Image.Channels, actual.Image.Height, actual.Image.Width))
        ?? Comparisons.Difference(expected.Image.Pixels, actual.Image.Pixels, ShadowTolerance)
        ?? Comparisons.Exact(string.Join(",", expected.Issues.Select(i => i.Kind)), string.Join(",", actual.Issues.Select(i => i.Kind)));

    // An app's drawer over the library's: each call falls back on its own, or is compared with the library's pixels.
    private sealed class GuardedDrawer(Slot slot, IDocumentPageDrawer app, IDocumentPageDrawer library) : IDocumentPageDrawer
    {
        public string Name => app.Name;

        public string Summary => app.Summary;

        public DocumentDrawing Draw(DocumentPage page, DocumentDrawOptions options) =>
            slot.Call(() => app.Draw(page, options), () => library.Draw(page, options), (a, b) => CompareDrawings(a, b));
    }
}
