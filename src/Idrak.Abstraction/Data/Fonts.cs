// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Data;

/// <summary>A font's vertical metrics, in font units (y up: the ascender is above the baseline, the descender below, negative).</summary>
/// <param name="UnitsPerEm">Font units per em.</param>
/// <param name="Ascender">Top of the tallest letters above the baseline.</param>
/// <param name="Descender">Bottom of the descenders (negative: below the baseline).</param>
/// <param name="LineGap">Extra space between lines.</param>
public readonly record struct FontMetrics(int UnitsPerEm, int Ascender, int Descender, int LineGap)
{
    /// <summary>The distance from one baseline to the next: ascender, minus descender, plus line gap.</summary>
    public int LineHeight => Ascender - Descender + LineGap;
}

/// <summary>What a step of an outline does; each takes its points from <see cref="GlyphOutline.Points"/> in order.</summary>
public enum OutlineVerb : byte
{
    /// <summary>Starts a contour at one point.</summary>
    Move,

    /// <summary>A straight line to one point.</summary>
    Line,

    /// <summary>A quadratic curve: a control point, then the end point.</summary>
    Quadratic,

    /// <summary>A cubic curve: two control points, then the end point.</summary>
    Cubic,

    /// <summary>Closes the contour (a line back to its start); no point.</summary>
    Close,
}

/// <summary>
/// A glyph's outline in font units (y up): contours of lines and curves, filled by the nonzero winding rule.
/// <see cref="Points"/> holds x, y pairs: one for a move or a line, two for a quadratic curve, three for a cubic.
/// </summary>
public sealed class GlyphOutline
{
    /// <summary>An outline with no contour (a space).</summary>
    public static readonly GlyphOutline Empty = new([], []);

    /// <summary>An outline of <paramref name="verbs"/> and their points (x, y pairs; used directly, not copied).</summary>
    /// <exception cref="ArgumentException">The points are not the verbs' count.</exception>
    public GlyphOutline(OutlineVerb[] verbs, float[] points)
    {
        ArgumentNullException.ThrowIfNull(verbs);
        ArgumentNullException.ThrowIfNull(points);
        int needed = 0;
        foreach (var verb in verbs)
        {
            needed += verb switch { OutlineVerb.Move or OutlineVerb.Line => 2, OutlineVerb.Quadratic => 4, OutlineVerb.Cubic => 6, _ => 0 };
        }

        if (needed != points.Length)
        {
            throw new ArgumentException($"The verbs take {needed / 2} points; {points.Length / 2.0} are given.", nameof(points));
        }

        Verbs = verbs;
        Points = points;
    }

    /// <summary>The steps of the contours.</summary>
    public ReadOnlyMemory<OutlineVerb> Verbs { get; }

    /// <summary>The steps' points, x then y.</summary>
    public ReadOnlyMemory<float> Points { get; }

    /// <summary>Whether there is nothing to fill.</summary>
    public bool IsEmpty => Verbs.Length == 0;

    /// <summary>The smallest box around the points (control points included), or all zero when empty.</summary>
    public (float MinX, float MinY, float MaxX, float MaxY) Bounds()
    {
        var p = Points.Span;
        if (p.Length == 0)
        {
            return default;
        }

        float minX = p[0], maxX = p[0], minY = p[1], maxY = p[1];
        for (int i = 2; i < p.Length; i += 2)
        {
            minX = MathF.Min(minX, p[i]);
            maxX = MathF.Max(maxX, p[i]);
            minY = MathF.Min(minY, p[i + 1]);
            maxY = MathF.Max(maxY, p[i + 1]);
        }

        return (minX, minY, maxX, maxY);
    }
}

/// <summary>
/// One font of a font file: its glyphs by character, their advances and outlines, and its raw tables (the shapers read the
/// layout tables, GSUB, GPOS and GDEF, from <see cref="Table"/>). Safe to use from several threads at once.
/// </summary>
public interface IFontFace
{
    /// <summary>The family name ("DejaVu Sans"), or the file's name when the font has none.</summary>
    string Family { get; }

    /// <summary>The style name ("Bold"), or "Regular".</summary>
    string Style { get; }

    /// <summary>The name of the <see cref="IFontFormat"/> that read it.</summary>
    string Format { get; }

    /// <summary>The vertical metrics.</summary>
    FontMetrics Metrics { get; }

    /// <summary>The number of glyphs (glyph ids are 0 to this minus one; 0 is the missing-character glyph).</summary>
    int GlyphCount { get; }

    /// <summary>The glyph of a Unicode code point, or 0 when the font has none.</summary>
    int Glyph(int codePoint);

    /// <summary>A glyph's advance width, in font units.</summary>
    int Advance(int glyph);

    /// <summary>A glyph's outline (<see cref="GlyphOutline.Empty"/> for a glyph with none).</summary>
    GlyphOutline Outline(int glyph);

    /// <summary>The bytes of the table <paramref name="tag"/> ("GSUB"), or empty when the font has none.</summary>
    ReadOnlyMemory<byte> Table(string tag);
}

/// <summary>Reads one kind of font file into its faces; register it with <see cref="FontFormats.Register"/>.</summary>
public interface IFontFormat
{
    /// <summary>The format's name, e.g. "opentype" (names ignore case).</summary>
    string Name { get; }

    /// <summary>The file extensions of the format, with the dot, e.g. ".ttf".</summary>
    IReadOnlyCollection<string> Extensions { get; }

    /// <summary>Whether a file starting with <paramref name="header"/> (its first bytes) is in this format.</summary>
    bool Accepts(ReadOnlySpan<byte> header);

    /// <summary>
    /// The faces of a whole file: one, or each of a collection. A damaged file is an <see cref="InvalidDataException"/>,
    /// a valid file in a variant the format does not read a <see cref="NotSupportedException"/>.
    /// </summary>
    /// <param name="file">The file's bytes (kept by the faces, not copied).</param>
    /// <param name="name">A name for messages and for a face whose font has no family name (the file's name).</param>
    IReadOnlyList<IFontFace> Read(ReadOnlyMemory<byte> file, string name);
}

/// <summary>
/// The font file formats, by name (ignoring case). Built in (Idrak.Imaging): "opentype" (.ttf, .otf, .ttc: TrueType
/// outlines, single fonts and collections). Register another with <see cref="Register"/>.
/// </summary>
public static class FontFormats
{
    /// <summary>The bytes of a file's start that <see cref="IFontFormat.Accepts"/> receives.</summary>
    public const int HeaderBytes = 16;

    private static readonly SlotTable<string, IFontFormat> Registry =
        new(nameof(FontFormats), (slot, app, library) => new GuardedFormat(slot, app, library), StringComparer.OrdinalIgnoreCase, newestFirst: true);

    static FontFormats() => LibraryDefaults.Ensure(typeof(FontFormats));

    /// <summary>Registers <paramref name="format"/> (one of a registered name takes its place; the library's stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IFontFormat format)
    {
        ArgumentNullException.ThrowIfNull(format);
        ArgumentException.ThrowIfNullOrWhiteSpace(format.Name);
        Registry.Register(format.Name, format, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's format <paramref name="name"/> (a built-in name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered format names, the first asked first.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The extensions of the registered formats (with the dot, lower case).</summary>
    public static IReadOnlyCollection<string> Extensions => [.. Registry.Values.SelectMany(f => f.Extensions).Select(e => e.ToLowerInvariant()).Distinct()];

    /// <summary>The format registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name.</exception>
    public static IFontFormat Get(string name) => Registry.Find(name)
        ?? throw new NotSupportedException($"No font format '{name}' is registered ({string.Join(", ", Names)}); add it with FontFormats.Register.");

    /// <summary>The library's format <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IFontFormat? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the format <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>What happens when the app's format <paramref name="name"/> fails.</summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Registry.SetPolicy(name, policy, shadowRate);

    /// <summary>The faces of a font file, read by the format that accepts its first bytes.</summary>
    /// <exception cref="InvalidDataException">No registered format reads the file, or it is damaged.</exception>
    public static IReadOnlyList<IFontFace> Read(string path) => Read(File.ReadAllBytes(path), Path.GetFileNameWithoutExtension(path));

    /// <summary>The faces of a font file held in memory (kept by the faces, not copied).</summary>
    /// <exception cref="InvalidDataException">No registered format reads the bytes, or they are damaged.</exception>
    public static IReadOnlyList<IFontFace> Read(ReadOnlyMemory<byte> file, string name = "font")
    {
        var header = file.Span[..Math.Min(file.Length, HeaderBytes)];
        foreach (var format in Registry.Values)
        {
            if (format.Accepts(header))
            {
                try
                {
                    return format.Read(file, name);
                }
                catch (Exception e) when (e is IndexOutOfRangeException or ArgumentOutOfRangeException or ArgumentException or EndOfStreamException)
                {
                    throw new InvalidDataException($"{name}: a damaged {format.Name} file ({e.Message}).", e);
                }
            }
        }

        throw new InvalidDataException($"{name}: not a font of the registered formats ({string.Join(", ", Names)}); register one with FontFormats.Register.");
    }

    /// <summary>The faces of every font file under <paramref name="folder"/> (and its subfolders) whose extension a format lists, in path order.</summary>
    public static IReadOnlyList<IFontFace> ReadFolder(string folder)
    {
        var extensions = Extensions;
        var faces = new List<IFontFace>();
        foreach (string path in Directory.EnumerateFiles(folder, "*", SearchOption.AllDirectories).Order(StringComparer.Ordinal))
        {
            if (extensions.Contains(Path.GetExtension(path).ToLowerInvariant()))
            {
                faces.AddRange(Read(path));
            }
        }

        return faces;
    }

    // An app's format over a built-in: each read falls back on its own, or is compared with the built-in's.
    private sealed class GuardedFormat(Slot slot, IFontFormat app, IFontFormat library) : IFontFormat
    {
        public string Name => app.Name;

        public IReadOnlyCollection<string> Extensions => app.Extensions;

        public bool Accepts(ReadOnlySpan<byte> header)
        {
            try
            {
                return app.Accepts(header);
            }
            catch (Exception e) when (slot.Failed(e))
            {
                return library.Accepts(header);
            }
        }

        public IReadOnlyList<IFontFace> Read(ReadOnlyMemory<byte> file, string name)
        {
            if (slot.Policy == SlotPolicy.Shadow)
            {
                var run = slot.Shadow();
                var answer = library.Read(file, name);
                if (run is not null)
                {
                    run.Answered();
                    try
                    {
                        var other = app.Read(file, name);
                        run.Done(Comparisons.Exact(Shape(answer), Shape(other)));
                    }
                    catch (Exception e) when (e is not OperationCanceledException)
                    {
                        run.Failed(e);
                    }
                }

                return answer;
            }

            try
            {
                return app.Read(file, name);
            }
            catch (Exception e) when (slot.Failed(e))
            {
                return library.Read(file, name);
            }
        }

        private static string Shape(IReadOnlyList<IFontFace> faces) => string.Join("; ", faces.Select(f => $"{f.Family} {f.Style} {f.GlyphCount} {f.Metrics}"));
    }
}

/// <summary>
/// Fills outlines into coverage: how much of each pixel the shape covers, 0 to 1. Register one with
/// <see cref="OutlineRasterizers.Register"/>.
/// </summary>
public interface IOutlineRasterizer
{
    /// <summary>The rasterizer's name (names ignore case).</summary>
    string Name { get; }

    /// <summary>
    /// Adds <paramref name="outline"/> to <paramref name="coverage"/> (<paramref name="height"/> rows of
    /// <paramref name="width"/>, row 0 at the top): the outline is scaled by <paramref name="scale"/> pixels per font
    /// unit, flipped so y grows downwards, and its origin placed at (<paramref name="x"/>, <paramref name="y"/>) pixels
    /// (the baseline at y). A pixel's coverage combines with what is there as ink over ink: 1 - (1 - a)(1 - b).
    /// </summary>
    void Fill(GlyphOutline outline, float scale, float x, float y, Span<float> coverage, int width, int height);
}

/// <summary>The outline rasterizers, by name. Built in (Idrak.Imaging): "coverage" (exact area coverage per pixel, nonzero winding, no hinting).</summary>
public static class OutlineRasterizers
{
    /// <summary>The library's rasterizer.</summary>
    public const string Coverage = "coverage";

    private static readonly SlotTable<string, IOutlineRasterizer> Registry =
        new(nameof(OutlineRasterizers), (slot, app, library) => new GuardedRasterizer(slot, app, library), StringComparer.OrdinalIgnoreCase);

    static OutlineRasterizers() => LibraryDefaults.Ensure(typeof(OutlineRasterizers));

    /// <summary>Registers <paramref name="rasterizer"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IOutlineRasterizer rasterizer)
    {
        ArgumentNullException.ThrowIfNull(rasterizer);
        ArgumentException.ThrowIfNullOrWhiteSpace(rasterizer.Name);
        Registry.Register(rasterizer.Name, rasterizer, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's rasterizer of that name (the library's, if any, answers again).</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered names.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The rasterizer of that name.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name.</exception>
    public static IOutlineRasterizer Get(string name = Coverage) => Registry.Find(name)
        ?? throw new NotSupportedException($"No outline rasterizer '{name}' is registered ({string.Join(", ", Names)}); add it with OutlineRasterizers.Register.");

    /// <summary>The library's rasterizer of that name, or null.</summary>
    public static IOutlineRasterizer? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the rasterizer of that name.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>What happens when the app's rasterizer of that name fails.</summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Registry.SetPolicy(name, policy, shadowRate);

    // An app's rasterizer over the library's: a failed fill is redone by the library's (on a copy, so a half fill is not kept).
    private sealed class GuardedRasterizer(Slot slot, IOutlineRasterizer app, IOutlineRasterizer library) : IOutlineRasterizer
    {
        public string Name => app.Name;

        public void Fill(GlyphOutline outline, float scale, float x, float y, Span<float> coverage, int width, int height)
        {
            if (slot.Policy == SlotPolicy.Shadow)
            {
                var run = slot.Shadow();
                float[]? before = run is null ? null : coverage.ToArray();
                library.Fill(outline, scale, x, y, coverage, width, height);
                if (run is not null)
                {
                    run.Answered();
                    try
                    {
                        app.Fill(outline, scale, x, y, before, width, height);
                        run.Done(Comparisons.Difference(coverage.ToArray(), before!, 1e-3f));
                    }
                    catch (Exception e) when (e is not OperationCanceledException)
                    {
                        run.Failed(e);
                    }
                }

                return;
            }

            float[]? copy = slot.Policy == SlotPolicy.FallBack ? coverage.ToArray() : null;   // only a fallback needs the coverage before
            try
            {
                app.Fill(outline, scale, x, y, coverage, width, height);
            }
            catch (Exception e) when (slot.Failed(e))                       // true only under FallBack, which made the copy
            {
                copy?.CopyTo(coverage);
                library.Fill(outline, scale, x, y, coverage, width, height);
            }
        }
    }
}
