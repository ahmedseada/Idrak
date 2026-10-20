// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Data;

namespace Idrak.Abstraction.Testing;

/// <summary>One line a segmenter under test found: its box on the (deskewed) page and its pixels.</summary>
/// <param name="X">The box's left edge.</param>
/// <param name="Y">The box's top edge.</param>
/// <param name="Width">The box's width.</param>
/// <param name="Height">The box's height.</param>
/// <param name="Image">The line's pixels.</param>
public sealed record SegmentedLine(int X, int Y, int Width, int Height, ImageData Image);

/// <summary>A page as a segmenter under test cut it: its lines top to bottom and the skew it undid (degrees).</summary>
/// <param name="Lines">The lines, top to bottom.</param>
/// <param name="Skew">The skew undone.</param>
public sealed record SegmentedPage(IReadOnlyList<SegmentedLine> Lines, double Skew);

/// <summary>
/// The checks of a text line segmenter (Idrak.Vision's <c>ITextLineSegmenter</c>, given as a delegate, a page in and its
/// lines out, since this kit depends on Idrak.Abstraction alone) on pages drawn here: lines of letter-like strokes of random
/// heights and lengths, some with dots above them, dark ink on light paper or light on dark, grey or colour, and blank pages.
/// <list type="bullet">
/// <item>a blank page has no lines;</item>
/// <item>every box lies on the page, its image is its box's size (one grey channel), boxes run top to bottom without
/// overlapping;</item>
/// <item>one line per drawn line, each drawn line's middle row inside its own box (dots joined to their line);</item>
/// <item>no skew found on a straight page;</item>
/// <item>the same lines again and from several threads at once;</item>
/// <item>given a <see cref="Reference"/> (the library's segmenter it replaces), the same boxes and skew.</item>
/// </list>
/// </summary>
/// <param name="reference">The library's segmenter to agree with, or null.</param>
public sealed class TextLineSegmenterSuite(Func<ImageData, SegmentedPage>? reference = null) : ContractSuite<Func<ImageData, SegmentedPage>>
{
    /// <summary>The library's segmenter to agree with, or null.</summary>
    public Func<ImageData, SegmentedPage>? Reference { get; } = reference;

    /// <inheritdoc />
    public override string Name => "text-line-segmenter";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        yield return Case("blank page", 1, 200, 160, 0, 14, dots: false, inverted: false, channels: 1);
        yield return Case("one line", 2, 240, 80, 1, 16, dots: false, inverted: false, channels: 1);
        yield return Case("four lines with dots", 3, 300, 260, 4, 14, dots: true, inverted: false, channels: 1);
        yield return Case("light ink on dark paper", 4, 260, 200, 3, 12, dots: false, inverted: true, channels: 1);
        yield return Case("a colour page", 5, 280, 220, 3, 16, dots: true, inverted: false, channels: 3);
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        int lineHeight = random.Next(10, large ? 40 : 22);
        int lines = random.Next(0, large ? 12 : 6);
        int width = random.Next(Math.Max(120, lineHeight * 6), large ? 1600 : 400);
        int height = Math.Max(lineHeight * 4, (int)(lines * lineHeight * 2.6) + 4 * lineHeight);
        return Case($"random page of {lines} lines", random.Next(), width, height, lines, lineHeight, random.Next(2) == 0, random.Next(4) == 0, random.Next(3) == 0 ? 3 : 1);
    }

    /// <inheritdoc />
    public override void Run(Func<ImageData, SegmentedPage> implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var (page, rows) = Draw(@case.Data);
        var found = implementation(page);
        var lines = found.Lines;
        checks.Check("boxes on the page, images their boxes' size", lines.All(l => l.X >= 0 && l.Y >= 0 && l.Width > 0 && l.Height > 0 && l.X + l.Width <= page.Width
                && l.Y + l.Height <= page.Height && l.Image.Width == l.Width && l.Image.Height == l.Height && l.Image.Channels == 1),
            string.Join("; ", lines.Select(l => $"({l.X}, {l.Y}) {l.Width} x {l.Height}, image {l.Image.Channels} x {l.Image.Height} x {l.Image.Width}")));
        checks.Check("top to bottom, not overlapping", lines.Zip(lines.Skip(1)).All(p => p.First.Y + p.First.Height <= p.Second.Y + p.Second.Height / 2 && p.First.Y < p.Second.Y),
            string.Join("; ", lines.Select(l => $"{l.Y}..{l.Y + l.Height}")));
        if (rows.Count == 0)
        {
            checks.Check("a blank page has no lines", lines.Count == 0, $"{lines.Count} lines on a blank page");
        }
        else
        {
            checks.Check("one line per drawn line", lines.Count == rows.Count, $"{lines.Count} lines for {rows.Count} drawn");
            if (lines.Count == rows.Count)
            {
                var missed = rows.Select((r, i) => (r, i)).Where(p => (p.r.Top + p.r.Bottom) / 2 is var middle && (middle < lines[p.i].Y || middle >= lines[p.i].Y + lines[p.i].Height))
                    .Select(p => $"line {p.i + 1} drawn {p.r.Top}..{p.r.Bottom}, found {lines[p.i].Y}..{lines[p.i].Y + lines[p.i].Height}").ToList();
                checks.Check("each drawn line inside its own box", missed.Count == 0, string.Join("; ", missed));
            }
        }

        checks.Check("no skew on a straight page", found.Skew == 0, $"skew {found.Skew}");
        checks.Compare("the same lines again", Different(found, implementation(page)));
        var parallel = new SegmentedPage[4];
        Parallel.For(0, parallel.Length, i => parallel[i] = implementation(page));
        checks.Compare("the same lines from several threads", parallel.Select(p => Different(found, p)).FirstOrDefault(d => d is not null));
        if (Reference is not null)
        {
            checks.Compare("agrees with the reference", Different(Reference(page), found));
        }
        else
        {
            checks.Skip("agrees with the reference", "no reference segmenter given");
        }
    }

    // How two pages' lines differ (boxes, pixels, skew), or null.
    private static string? Different(SegmentedPage expected, SegmentedPage actual)
    {
        if (expected.Lines.Count != actual.Lines.Count)
        {
            return $"{actual.Lines.Count} lines, expected {expected.Lines.Count}";
        }

        for (int i = 0; i < expected.Lines.Count; i++)
        {
            var (e, a) = (expected.Lines[i], actual.Lines[i]);
            if (e.X != a.X || e.Y != a.Y || e.Width != a.Width || e.Height != a.Height)
            {
                return $"line {i + 1} at ({a.X}, {a.Y}) {a.Width} x {a.Height}, expected ({e.X}, {e.Y}) {e.Width} x {e.Height}";
            }

            if (!e.Image.Pixels.AsSpan().SequenceEqual(a.Image.Pixels))
            {
                return $"line {i + 1}: other pixels";
            }
        }

        return expected.Skew == actual.Skew ? null : $"skew {actual.Skew}, expected {expected.Skew}";
    }

    // The page a case draws: its lines of strokes (ink rows [Top, Bottom) of each line's body), dots a few rows above some.
    private static (ImageData Page, List<(int Top, int Bottom)> Rows) Draw(JsonObject data)
    {
        var random = new Random((int)data["seed"]!);
        int width = (int)data["width"]!, height = (int)data["height"]!, count = (int)data["lines"]!, lineHeight = (int)data["line_height"]!, channels = (int)data["channels"]!;
        bool dots = (bool)data["dots"]!, inverted = (bool)data["inverted"]!;
        float paper = inverted ? 0.08f : 0.94f, ink = inverted ? 0.9f : 0.12f;
        var plane = new float[width * height];
        Array.Fill(plane, paper);
        var rows = new List<(int Top, int Bottom)>();
        int pitch = count == 0 ? 0 : (height - 2 * lineHeight) / count;
        for (int l = 0; l < count; l++)
        {
            int top = lineHeight + l * pitch + (pitch - lineHeight) / 2;
            int x = random.Next(2, Math.Max(3, width / 10));
            int end = width - random.Next(2, Math.Max(3, width / 10));
            while (x < end - 4)
            {
                // A word: strokes of the line's height with a gap of a stroke, then a space.
                int letters = random.Next(2, 7);
                for (int k = 0; k < letters && x < end - 3; k++)
                {
                    int stroke = random.Next(2, Math.Max(3, lineHeight / 3));
                    Fill(plane, width, x, top, stroke, lineHeight, ink);
                    if (dots && random.Next(3) == 0 && top >= 5)
                    {
                        Fill(plane, width, x, top - 4, 2, 2, ink);
                    }

                    x += stroke + random.Next(1, 4);
                }

                x += random.Next(lineHeight / 2, lineHeight + 1);
            }

            rows.Add((top, top + lineHeight));
        }

        var pixels = new float[channels * plane.Length];
        for (int c = 0; c < channels; c++)
        {
            plane.CopyTo(pixels, c * plane.Length);
        }

        return (new ImageData(pixels, channels, height, width), rows);
    }

    private static void Fill(float[] plane, int width, int x, int y, int w, int h, float value)
    {
        for (int r = y; r < y + h; r++)
        {
            plane.AsSpan(r * width + x, Math.Min(w, width - x)).Fill(value);
        }
    }

    private static ContractCase Case(string name, int seed, int width, int height, int lines, int lineHeight, bool dots, bool inverted, int channels) =>
        new(name, new JsonObject
        {
            ["seed"] = seed, ["width"] = width, ["height"] = height, ["lines"] = lines, ["line_height"] = lineHeight, ["dots"] = dots, ["inverted"] = inverted,
            ["channels"] = channels,
        });
}
