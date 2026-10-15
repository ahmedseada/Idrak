// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Data;

/// <summary>
/// What a page says, by part: the header's three places, a title, a subject line, the body's paragraphs, a signature
/// block and a footer. "Start" and "end" follow the text's direction (start is the right side of an Arabic page). Empty
/// parts are not drawn.
/// </summary>
public sealed record PageContent
{
    /// <summary>Header lines at the start side (an Arabic letter's state and ministry lines, at its right).</summary>
    public IReadOnlyList<string> HeaderStart { get; init; } = [];

    /// <summary>Header lines in the middle (a basmala, an emblem's caption).</summary>
    public IReadOnlyList<string> HeaderCenter { get; init; } = [];

    /// <summary>Header lines at the end side (number, date, attachments).</summary>
    public IReadOnlyList<string> HeaderEnd { get; init; } = [];

    /// <summary>A title over the body, centred; null for none.</summary>
    public string? Title { get; init; }

    /// <summary>A subject line; null for none.</summary>
    public string? Subject { get; init; }

    /// <summary>The body's paragraphs.</summary>
    public IReadOnlyList<string> Body { get; init; } = [];

    /// <summary>The signature block's lines (a name and a title), at the end side.</summary>
    public IReadOnlyList<string> Signature { get; init; } = [];

    /// <summary>Footer lines, under a rule at the bottom.</summary>
    public IReadOnlyList<string> Footer { get; init; } = [];

    /// <summary>The text's language, a BCP 47 tag ("ar"), or null.</summary>
    public string? Language { get; init; }

    /// <summary>Whether the page reads right to left; null to follow its text.</summary>
    public bool? RightToLeft { get; init; }

    /// <summary>The page's text in reading order, one part a line, as a page drawn from it reads (header start, centre, end; title; subject; body; signature; footer).</summary>
    public string ReadingOrder() => string.Join("\n", HeaderStart.Concat(HeaderCenter).Concat(HeaderEnd)
        .Concat(Title is null ? [] : [Title]).Concat(Subject is null ? [] : [Subject]).Concat(Body).Concat(Signature).Concat(Footer).Where(s => s.Length > 0));
}

/// <summary>
/// How a page is drawn: its fonts (one is drawn per page, others back up characters it lacks), its size, and the ranges a
/// layout draws its sizes and spacing from (each page its own values, from its seed).
/// </summary>
public sealed record PageStyle
{
    /// <summary>The fonts a page's text is set in (one drawn per page; per part for some layouts).</summary>
    public required IReadOnlyList<IFontFace> Fonts { get; init; }

    /// <summary>Fonts for characters a page's font lacks (Latin, digits), tried in order.</summary>
    public IReadOnlyList<IFontFace> Fallbacks { get; init; } = [];

    /// <summary>The page's width in pixels (1240: A4 at 150 dots per inch).</summary>
    public int Width { get; init; } = 1240;

    /// <summary>The page's least height in pixels (1754: A4 at 150 dots per inch); a longer text makes a taller page.</summary>
    public int Height { get; init; } = 1754;

    /// <summary>The body's text size range, in pixels per em.</summary>
    public (float Min, float Max) BodySize { get; init; } = (26, 36);

    /// <summary>The line spacing range (multiples of the font's line height).</summary>
    public (float Min, float Max) LineSpacing { get; init; } = (1.0f, 1.5f);

    /// <summary>The side margins, as a share of the width.</summary>
    public (float Min, float Max) Margin { get; init; } = (0.05f, 0.10f);

    /// <summary>How often the body is justified (0 to 1); else aligned at its start.</summary>
    public float Justify { get; init; } = 0.5f;

    /// <summary>How often a letter's header is set in another of the fonts than its body (0 to 1).</summary>
    public float OtherHeaderFont { get; init; } = 0.4f;

    /// <summary>How often a letter has a rule under its header (0 to 1).</summary>
    public float Rule { get; init; } = 0.7f;

    /// <summary>How often a letter with no centre header lines has an emblem's ring there (0 to 1).</summary>
    public float Emblem { get; init; } = 0.5f;

    /// <summary>How often a letter with a signature has a stamp's ring over it (0 to 1).</summary>
    public float Stamp { get; init; } = 0.3f;

    /// <summary>How often a colour letter's signature is in blue ink (0 to 1).</summary>
    public float BlueSignature { get; init; } = 0.4f;

    /// <summary>The ink's darkness range (0 paper, 1 black).</summary>
    public (float Min, float Max) Ink { get; init; } = (0.8f, 1.0f);

    /// <summary>Colour pages: tinted paper, blue signatures and stamps (else grey).</summary>
    public bool Color { get; init; } = true;

    /// <summary>
    /// How often a page is cut to its drawn parts (0 to 1; 0, the default: the whole page): the box around every part,
    /// with <see cref="CropPadding"/> of paper around it, so the text fills the image (for a reader learning from
    /// scratch, the text and not the paper). The text is unchanged: every drawn part is inside the box.
    /// </summary>
    public float Crop { get; init; }

    /// <summary>The paper kept around a cut page's parts, in pixels (a range: each page its own).</summary>
    public (float Min, float Max) CropPadding { get; init; } = (8, 24);
}

/// <summary>A drawn part of a page: what it is, its text and its box (pixels).</summary>
/// <param name="Kind">"header", "title", "subject", "body", "signature", "footer", "stamp" or "rule".</param>
/// <param name="Text">Its text in reading order (empty for a drawing).</param>
/// <param name="X">Its left edge.</param>
/// <param name="Y">Its top edge.</param>
/// <param name="Width">Its width.</param>
/// <param name="Height">Its height.</param>
public readonly record struct PageBlock(string Kind, string Text, float X, float Y, float Width, float Height);

/// <summary>A page drawn by a layout: its image, its text in reading order (what an OCR of it should read), and its parts.</summary>
/// <param name="Image">The page, red, green and blue (or grey) in [0, 1].</param>
/// <param name="Text">The drawn text in reading order, one part a line (paragraphs whole, not broken as drawn).</param>
/// <param name="Blocks">The drawn parts in reading order.</param>
/// <param name="Description">What was chosen (font, sizes, alignment), for a data set's manifest.</param>
public sealed record RenderedPage(ImageData Image, string Text, IReadOnlyList<PageBlock> Blocks, string Description);

/// <summary>Draws a page's content in one arrangement (a letter, a plain page); register one with <see cref="PageLayouts.Register"/>.</summary>
public interface IPageLayout
{
    /// <summary>The layout's name ("letter").</summary>
    string Name { get; }

    /// <summary>One line on what it draws.</summary>
    string Summary { get; }

    /// <summary>Draws <paramref name="content"/> in <paramref name="style"/>, the choices within the style's ranges made from <paramref name="seed"/> (the same seed, the same page).</summary>
    RenderedPage Render(PageContent content, PageStyle style, int seed);
}

/// <summary>
/// The page layouts, by name. Built in (Idrak.Imaging): "letter" (an official letter: a header of three places, a rule,
/// subject, body, signature block, an optional stamp, footer) and "plain" (header lines, title, body, footer, no ornament).
/// Register another with <see cref="Register"/>.
/// </summary>
public static class PageLayouts
{
    private static readonly SlotTable<string, IPageLayout> Registry =
        new(nameof(PageLayouts), (slot, app, library) => new GuardedLayout(slot, app, library), StringComparer.OrdinalIgnoreCase);

    static PageLayouts() => LibraryDefaults.Ensure(typeof(PageLayouts));

    /// <summary>Registers <paramref name="layout"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IPageLayout layout)
    {
        ArgumentNullException.ThrowIfNull(layout);
        ArgumentException.ThrowIfNullOrWhiteSpace(layout.Name);
        Registry.Register(layout.Name, layout, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's layout of that name (the library's, if any, answers again).</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered names.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The layout of that name.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name.</exception>
    public static IPageLayout Get(string name) => Registry.Find(name)
        ?? throw new NotSupportedException($"No page layout '{name}' is registered ({string.Join(", ", Names)}); add it with PageLayouts.Register.");

    /// <summary>The library's layout of that name, or null.</summary>
    public static IPageLayout? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the layout of that name.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>What happens when the app's layout of that name fails.</summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Registry.SetPolicy(name, policy, shadowRate);

    private sealed class GuardedLayout(Slot slot, IPageLayout app, IPageLayout library) : IPageLayout
    {
        public string Name => app.Name;

        public string Summary => app.Summary;

        public RenderedPage Render(PageContent content, PageStyle style, int seed)
        {
            if (slot.Policy == SlotPolicy.Shadow)
            {
                var run = slot.Shadow();
                var answer = library.Render(content, style, seed);
                if (run is not null)
                {
                    run.Answered();
                    try
                    {
                        var other = app.Render(content, style, seed);
                        run.Done(Comparisons.Exact(answer.Text, other.Text));
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
                return app.Render(content, style, seed);
            }
            catch (Exception e) when (slot.Failed(e))
            {
                return library.Render(content, style, seed);
            }
        }
    }
}
