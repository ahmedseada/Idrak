// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Vision.Abstractions;

/// <summary>A text line found on a page: its place (in the deskewed page's pixels) and its pixels (grey, as on the page).</summary>
/// <param name="Index">1 for the top line, then down the page.</param>
/// <param name="X">The line's left edge (padding included).</param>
/// <param name="Y">The line's top edge (padding included).</param>
/// <param name="Width">Its width in pixels.</param>
/// <param name="Height">Its height in pixels.</param>
/// <param name="Image">Its pixels: one grey channel, cut from the deskewed page.</param>
public sealed record PageLine(int Index, int X, int Y, int Width, int Height, ImageData Image);

/// <summary>The lines of a page, top to bottom, and the skew undone before they were cut.</summary>
/// <param name="Lines">The lines, top to bottom.</param>
/// <param name="Skew">The skew undone, in degrees (0 when none was).</param>
public sealed record PageLines(IReadOnlyList<PageLine> Lines, double Skew);

/// <summary>
/// How lines are found (<see cref="TextLineSegmenters"/>). Every size is relative to the page or the line except
/// <see cref="MinPadding"/>; each segmenter reads those it has (the library's "projection" reads them all), and one of
/// its own reads <see cref="Parameters"/>.
/// </summary>
public sealed record LineSegmentationOptions
{
    /// <summary>The largest skew corrected, in degrees either way (0: no deskew). Default 3.</summary>
    public double MaxSkew { get; init; } = 3;

    /// <summary>The step between the skews tried, in degrees. Default 0.2.</summary>
    public double SkewStep { get; init; } = 0.2;

    /// <summary>
    /// How much sharper (the sum of squared row counts) a skewed profile must be than the straight one to be taken, as a
    /// factor. Default 1.02.
    /// </summary>
    public double MinSkewGain { get; init; } = 1.02;

    /// <summary>The most ink pixels the skew is measured on (every n-th ink pixel when the page has more). Default 200,000.</summary>
    public int MaxSkewSamples { get; init; } = 200_000;

    /// <summary>Padding around a line, as a share of its height. Default 0.2.</summary>
    public double Padding { get; init; } = 0.2;

    /// <summary>The least padding around a line, in pixels. Default 2.</summary>
    public int MinPadding { get; init; } = 2;

    /// <summary>A row has ink when at least this share of the page's width is ink (and at least one pixel). Default 0.001.</summary>
    public double RowInk { get; init; } = 0.001;

    /// <summary>A band with less ink than this share of a typical line's is a speck and left out. Default 0.05.</summary>
    public double SpeckShare { get; init; } = 0.05;

    /// <summary>
    /// A band shorter than this share of a typical line's height (dots, marks, a broken stroke) joins its nearer neighbour
    /// when the gap is under a typical line's height. Default 0.5.
    /// </summary>
    public double MarkShare { get; init; } = 0.5;

    /// <summary>Settings of a segmenter of one's own, by name (empty for the library's).</summary>
    public IReadOnlyDictionary<string, double> Parameters { get; init; } = new Dictionary<string, double>();
}

/// <summary>
/// Cuts a page image into its text lines: each line's place and pixels, top to bottom, with the page's skew undone. The
/// library's is "projection" (<see cref="ProjectionLineSegmenter"/>); register another with
/// <see cref="TextLineSegmenters.Register"/> (a learned layout model, a column-aware one), and readers and apps use it by
/// name.
/// </summary>
public interface ITextLineSegmenter
{
    /// <summary>Its name in <see cref="TextLineSegmenters"/>.</summary>
    string Name { get; }

    /// <summary>The lines of <paramref name="page"/>, top to bottom, and the skew undone (degrees); no lines for a blank page.</summary>
    PageLines Find(ImageData page, LineSegmentationOptions options);
}

/// <summary>
/// The text line segmenters, by name (ignoring case). The library's: <see cref="Projection"/> (the page binarized, its skew
/// measured by the sharpest row profile and undone, rows with ink grouped into bands, marks joined to their line, each band
/// cut out with padding). Register another with <see cref="Register"/>; one under the library's name takes its place, which
/// stays behind it (see <see cref="SetPolicy"/>: an app's segmenter may fall back on the library's per page, or run beside
/// it and be compared).
/// </summary>
public static class TextLineSegmenters
{
    /// <summary>"projection": lines by horizontal projection of the binarized, deskewed page.</summary>
    public const string Projection = "projection";

    private static readonly SlotTable<string, ITextLineSegmenter> Table = new(nameof(TextLineSegmenters), Guard, StringComparer.OrdinalIgnoreCase);

    static TextLineSegmenters() => Overrides.AsLibraryDefaults(() => Register(new ProjectionLineSegmenter()));

    /// <summary>Registers <paramref name="segmenter"/> under its <see cref="ITextLineSegmenter.Name"/>; under the library's name it takes its place.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ITextLineSegmenter segmenter)
    {
        ArgumentNullException.ThrowIfNull(segmenter);
        ArgumentException.ThrowIfNullOrWhiteSpace(segmenter.Name);
        Table.Register(segmenter.Name, segmenter, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's segmenter <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The segmenter <paramref name="name"/> (an app's guarded by its policy), or null.</summary>
    public static ITextLineSegmenter? Find(string name) => Table.TryGet(name, out var segmenter) ? segmenter : null;

    /// <summary>The segmenter <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names the registered ones.</exception>
    public static ITextLineSegmenter Get(string name) => Find(name)
        ?? throw new NotRegisteredException(nameof(TextLineSegmenters), name, Table.Keys,
            $"Text line segmenter '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); add it with TextLineSegmenters.Register.");

    /// <summary>The library's segmenter <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static ITextLineSegmenter? Default(string name) => Table.Default(name);

    /// <summary>Who registered the segmenter <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each registered segmenter as "name: what it does", one per line.</summary>
    public static string Describe() => string.Join("\n", Table.Keys.Select(k => $"{k}: {Table.Find(k)}"));

    /// <summary>
    /// What happens when the app's segmenter <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set;
    /// <see cref="SlotPolicy.FallBack"/> cuts the page with the library's), or whether it only runs beside the library's
    /// (<see cref="SlotPolicy.Shadow"/>: the library's lines are used, the two are compared).
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);

    /// <summary>The lines of <paramref name="page"/> by the segmenter <paramref name="name"/> (<see cref="Projection"/> unless given).</summary>
    /// <exception cref="NotSupportedException">No segmenter has that name.</exception>
    public static PageLines Segment(ImageData page, LineSegmentationOptions? options = null, string name = Projection)
    {
        ArgumentNullException.ThrowIfNull(page);
        return Get(name).Find(page, options ?? new LineSegmentationOptions());
    }

    private static ITextLineSegmenter Guard(Slot slot, ITextLineSegmenter app, ITextLineSegmenter library) => new GuardedSegmenter(slot, app, library);

    // An app's segmenter over the library's: each page falls back on its own, or is compared with the library's lines.
    private sealed class GuardedSegmenter(Slot slot, ITextLineSegmenter app, ITextLineSegmenter library) : ITextLineSegmenter
    {
        public string Name => app.Name;

        public PageLines Find(ImageData page, LineSegmentationOptions options) =>
            slot.Call(() => app.Find(page, options), () => library.Find(page, options), Difference);

        public override string ToString() => app.ToString() ?? app.Name;

        private static string? Difference(PageLines a, PageLines b)
        {
            if (a.Lines.Count != b.Lines.Count)
            {
                return $"{a.Lines.Count} lines, the library's {b.Lines.Count}";
            }

            for (int i = 0; i < a.Lines.Count; i++)
            {
                var (x, y) = (a.Lines[i], b.Lines[i]);
                if (x.X != y.X || x.Y != y.Y || x.Width != y.Width || x.Height != y.Height)
                {
                    return $"line {i + 1} at ({x.X}, {x.Y}) {x.Width} x {x.Height}, the library's at ({y.X}, {y.Y}) {y.Width} x {y.Height}";
                }
            }

            return a.Skew == b.Skew ? null : $"skew {a.Skew}, the library's {b.Skew}";
        }
    }
}
