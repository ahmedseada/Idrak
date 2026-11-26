// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Data;

namespace Idrak.Abstraction.Testing;

/// <summary>A glyph a sample image is known to hold: the character, its joining form and its box in the image.</summary>
/// <param name="Text">The character.</param>
/// <param name="Form">Its joining form, or null for none.</param>
/// <param name="Box">Its ink's box in the image, in pixels.</param>
public readonly record struct ExpectedGlyph(string Text, string? Form, PixelBox Box);

/// <summary>
/// An image a glyph source is checked on (<see cref="GlyphSourceSuite"/>): the image, its known text where the source
/// reads one, the options the source is read with, the glyphs it is known to hold where they are known, and the paper's
/// value (what the suite pads and blanks the image with).
/// </summary>
/// <param name="Image">The image.</param>
/// <param name="Text">Its known text (its transcription), or null for a source that knows its labels otherwise.</param>
public sealed record GlyphSourceSample(ImageData Image, string? Text = null)
{
    /// <summary>The image.</summary>
    public ImageData Image { get; init; } = Image ?? throw new ArgumentNullException(nameof(Image));

    /// <summary>A name for reports.</summary>
    public string Name { get; init; } = "a sample";

    /// <summary>The options the source is read with (its own keys); null for its defaults.</summary>
    public IReadOnlyDictionary<string, string>? Options { get; init; }

    /// <summary>The glyphs the image is known to hold, where known: each must be found with its label, around its box.</summary>
    public IReadOnlyList<ExpectedGlyph>? Glyphs { get; init; }

    /// <summary>The paper's value in every channel, from 0 to 1 (1: white), which the suite pads and blanks the image with.</summary>
    public float Paper { get; init; } = 1f;
}

/// <summary>
/// The checks of a glyph source (<see cref="IGlyphSource"/>, registered in <see cref="GlyphSources"/>) on images of its
/// own (<see cref="GlyphSourceSample"/>), each as given, padded with paper by random margins, and blank:
/// <list type="bullet">
/// <item>every glyph lies on the image with ink, its region on the image too;</item>
/// <item>every glyph lies inside its cell or its ink piece (<see cref="LabelledGlyph.Region"/>);</item>
/// <item>every glyph is labelled once: no two glyphs give the same ink (boxes overlapping by more than half the smaller);</item>
/// <item>the known text's characters (its text elements, spaces aside) are all accounted for: each given as a glyph or
/// reported left out with a reason, and nothing given that the text does not hold;</item>
/// <item>the glyphs the sample is known to hold are found with their label and form, around their box (moved with the
/// padding);</item>
/// <item>a page with no ink gives no glyph;</item>
/// <item>the same glyphs again and from several threads at once;</item>
/// <item>an option it does not take is refused, naming it.</item>
/// </list>
/// </summary>
public sealed class GlyphSourceSuite : ContractSuite<IGlyphSource>
{
    private readonly IReadOnlyList<GlyphSourceSample> _samples;

    /// <summary>The checks on <paramref name="samples"/>.</summary>
    /// <param name="samples">Images in the source's kind (at least one), with their known text and glyphs where known.</param>
    /// <exception cref="ArgumentException">No sample, or a paper value not from 0 to 1.</exception>
    public GlyphSourceSuite(IReadOnlyList<GlyphSourceSample> samples)
    {
        ArgumentNullException.ThrowIfNull(samples);
        if (samples.Count == 0)
        {
            throw new ArgumentException("A glyph source is checked on images of its own kind: give at least one.", nameof(samples));
        }

        foreach (var sample in samples)
        {
            ArgumentNullException.ThrowIfNull(sample, nameof(samples));
            if (!(sample.Paper is >= 0f and <= 1f))
            {
                throw new ArgumentException($"{sample.Name}: the paper's value is from 0 to 1, not {sample.Paper}.", nameof(samples));
            }
        }

        _samples = samples;
    }

    /// <summary>The option no glyph source takes, which each must refuse naming it.</summary>
    public const string UnknownOption = "glyph-source-suite-unknown-option";

    /// <inheritdoc />
    public override string Name => "glyph-source";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        for (int i = 0; i < _samples.Count; i++)
        {
            yield return Case(_samples[i].Name, i, 0, 0, 0, 0, blank: false);
            yield return Case($"{_samples[i].Name}, blank", i, 0, 0, 0, 0, blank: true);
        }
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        int i = random.Next(_samples.Count);
        var image = _samples[i].Image;
        int most = Math.Max(2, (large ? 20 : 8) * Math.Min(image.Width, image.Height) / 100);
        return Case($"{_samples[i].Name}, padded", i, random.Next(most + 1), random.Next(most + 1), random.Next(most + 1), random.Next(most + 1), blank: false);
    }

    /// <inheritdoc />
    public override void Run(IGlyphSource implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var data = @case.Data;
        var sample = _samples[(int)data["sample"]!];
        int left = (int)data["left"]!, top = (int)data["top"]!;
        bool blank = (bool)data["blank"]!;
        var image = Prepare(sample, left, top, (int)data["right"]!, (int)data["bottom"]!, blank);
        CheckRefusal(implementation, checks);
        var settings = implementation.Read(sample.Options ?? PageOptions.None);
        var request = new GlyphSourceRequest(image, sample.Text, settings);
        var found = implementation.Find(request);
        var glyphs = found.Glyphs;

        var off = glyphs.FirstOrDefault(g => !Inside(g.Box, image) || !Inside(g.Region, image) || g.Ink.IsBlank);
        checks.Check("every glyph on the image, with ink, its region on the image", off is null,
            $"{off} on an image of {image.Width} x {image.Height}{(off is { Ink.IsBlank: true } ? ", with no ink" : "")}");
        var outside = glyphs.FirstOrDefault(g => !Within(g.Box, g.Region));
        checks.Check("every glyph inside its cell or its ink piece", outside is null, $"{outside}: its box reaches outside its region");
        checks.Compare("every glyph labelled once", Twice(glyphs));

        if (sample.Text is { } text)
        {
            checks.Compare("the known text's characters all accounted for or reported", Accounted(text, found));
        }
        else
        {
            checks.Skip("the known text's characters all accounted for or reported", "the sample has no known text");
        }

        if (blank)
        {
            checks.Check("a page with no ink gives no glyph", glyphs.Count == 0, $"{glyphs.Count} glyphs on a blank page: {string.Join("; ", glyphs.Take(3))}");
        }
        else if (sample.Glyphs is { Count: > 0 } expected)
        {
            checks.Compare("the glyphs expected found where they are", Missing(expected, glyphs, left, top));
        }
        else
        {
            checks.Skip("the glyphs expected found where they are", "the sample does not say where its glyphs are");
        }

        string? again = Different(found, implementation.Find(request));
        var parallel = new FoundGlyphs[4];
        Parallel.For(0, parallel.Length, k => parallel[k] = implementation.Find(request));
        checks.Compare("the same glyphs again and from several threads", again ?? parallel.Select(p => Different(found, p)).FirstOrDefault(d => d is not null));
    }

    private static void CheckRefusal(IGlyphSource implementation, CaseChecks checks)
    {
        try
        {
            implementation.Read(new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase) { [UnknownOption] = "1" });
            checks.Fail("an option it does not take is refused", $"'{UnknownOption}' was taken");
        }
        catch (ArgumentException e)
        {
            checks.Check("an option it does not take is refused", e.Message.Contains(UnknownOption, StringComparison.Ordinal), $"the refusal does not name the option: {e.Message}");
        }
    }

    private static bool Inside(PixelBox box, ImageData image) => box.Width > 0 && box.Height > 0 && box.X >= 0 && box.Y >= 0 && box.Right <= image.Width && box.Bottom <= image.Height;

    private static bool Within(PixelBox box, PixelBox region) => box.X >= region.X && box.Y >= region.Y && box.Right <= region.Right && box.Bottom <= region.Bottom;

    // Two glyphs giving the same ink (boxes overlapping by more than half the smaller), or null.
    private static string? Twice(IReadOnlyList<LabelledGlyph> glyphs)
    {
        for (int a = 0; a < glyphs.Count; a++)
        {
            for (int b = a + 1; b < glyphs.Count; b++)
            {
                var (p, q) = (glyphs[a].Box, glyphs[b].Box);
                long w = Math.Min(p.Right, q.Right) - Math.Max(p.X, q.X), h = Math.Min(p.Bottom, q.Bottom) - Math.Max(p.Y, q.Y);
                if (w > 0 && h > 0 && 2 * w * h > Math.Min((long)p.Width * p.Height, (long)q.Width * q.Height))
                {
                    return $"glyphs {a + 1} and {b + 1} give the same ink: {glyphs[a]} and {glyphs[b]}";
                }
            }
        }

        return null;
    }

    // Whether the known text's characters (text elements, spaces aside) are exactly the glyphs' and the left-out ones'.
    private static string? Accounted(string text, FoundGlyphs found)
    {
        var expected = Count([text]);
        var given = Count([.. found.Glyphs.Select(g => g.Text), .. found.LeftOut.Select(l => l.Text)]);
        foreach (var (element, count) in expected)
        {
            int had = given.GetValueOrDefault(element);
            if (had != count)
            {
                return $"'{element}': {count} in the text, {had} given as glyphs or reported left out";
            }
        }

        var invented = given.Keys.FirstOrDefault(e => !expected.ContainsKey(e));
        if (invented is not null)
        {
            return $"'{invented}' is given {given[invented]} times, the text has none";
        }

        var silent = found.LeftOut.FirstOrDefault(l => string.IsNullOrWhiteSpace(l.Reason));
        return silent.Text is null ? null : $"'{silent.Text}' is left out without a reason";
    }

    // How many times each text element (spaces aside) appears in the texts.
    private static Dictionary<string, int> Count(IEnumerable<string> texts)
    {
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (string text in texts)
        {
            var elements = StringInfo.GetTextElementEnumerator(text ?? "");
            while (elements.MoveNext())
            {
                string element = elements.GetTextElement();
                if (!string.IsNullOrWhiteSpace(element))
                {
                    counts[element] = counts.GetValueOrDefault(element) + 1;
                }
            }
        }

        return counts;
    }

    // The first expected glyph no glyph answers (the same text and form, its box around the expected box's centre), or null.
    private static string? Missing(IReadOnlyList<ExpectedGlyph> expected, IReadOnlyList<LabelledGlyph> glyphs, int left, int top)
    {
        var used = new bool[glyphs.Count];
        foreach (var e in expected)
        {
            int cx = e.Box.X + left + e.Box.Width / 2, cy = e.Box.Y + top + e.Box.Height / 2;
            int match = -1;
            for (int g = 0; g < glyphs.Count && match < 0; g++)
            {
                var b = glyphs[g].Box;
                if (!used[g] && glyphs[g].Text == e.Text && glyphs[g].Form == e.Form && cx >= b.X && cx < b.Right && cy >= b.Y && cy < b.Bottom)
                {
                    match = g;
                }
            }

            if (match < 0)
            {
                return $"'{e.Text}'{(e.Form is null ? "" : $" ({e.Form})")} around ({cx}, {cy}) was not found; found: {string.Join("; ", glyphs.Take(6))}";
            }

            used[match] = true;
        }

        return null;
    }

    // How two readings differ (labels, forms, boxes, regions, baselines, sizes, ink; the characters left out), or null.
    private static string? Different(FoundGlyphs expected, FoundGlyphs actual)
    {
        if (expected.Glyphs.Count != actual.Glyphs.Count)
        {
            return $"{actual.Glyphs.Count} glyphs, the first time {expected.Glyphs.Count}";
        }

        for (int i = 0; i < expected.Glyphs.Count; i++)
        {
            var (e, a) = (expected.Glyphs[i], actual.Glyphs[i]);
            if (e.Text != a.Text || e.Form != a.Form || e.Box != a.Box || e.Region != a.Region || e.Baseline != a.Baseline || e.Size != a.Size)
            {
                return $"glyph {i + 1}: {a}, the first time {e}";
            }

            if (!e.Ink.Values.Span.SequenceEqual(a.Ink.Values.Span))
            {
                return $"glyph {i + 1} ('{a.Text}'): other ink";
            }
        }

        return expected.LeftOut.SequenceEqual(actual.LeftOut) ? null
            : $"left out: {string.Join("; ", actual.LeftOut.Select(l => l.Text))}, the first time {string.Join("; ", expected.LeftOut.Select(l => l.Text))}";
    }

    // The sample's image padded with paper, or a page of paper alone of the same size.
    private static ImageData Prepare(GlyphSourceSample sample, int left, int top, int right, int bottom, bool blank)
    {
        var image = sample.Image;
        if (!blank && left == 0 && top == 0 && right == 0 && bottom == 0)
        {
            return image;
        }

        int width = image.Width + left + right, height = image.Height + top + bottom, channels = image.Channels;
        var pixels = new float[channels * width * height];
        Array.Fill(pixels, sample.Paper);
        if (blank)
        {
            return new ImageData(pixels, channels, height, width);
        }

        var source = image.Pixels;
        for (int c = 0; c < channels; c++)
        {
            for (int y = 0; y < image.Height; y++)
            {
                source.AsSpan((c * image.Height + y) * image.Width, image.Width).CopyTo(pixels.AsSpan((c * height + y + top) * width + left, image.Width));
            }
        }

        return new ImageData(pixels, channels, height, width);
    }

    private static ContractCase Case(string name, int sample, int left, int top, int right, int bottom, bool blank) =>
        new(name, new JsonObject { ["sample"] = sample, ["left"] = left, ["top"] = top, ["right"] = right, ["bottom"] = bottom, ["blank"] = blank });
}
