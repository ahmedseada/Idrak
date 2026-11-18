// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Data;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// Glyphs a font maker is checked on (<see cref="FontMakerSuite"/>): what a glyph source found (labelled glyphs with their
/// ink, box, baseline and size), the options the maker is read with, and the tracer the glyphs are traced with.
/// </summary>
/// <param name="Glyphs">The labelled glyphs (a sheet's worth, say), and what their source left out.</param>
public sealed record FontMakerSample(FoundGlyphs Glyphs)
{
    /// <summary>The labelled glyphs, and what their source left out.</summary>
    public FoundGlyphs Glyphs { get; init; } = Glyphs ?? throw new ArgumentNullException(nameof(Glyphs));

    /// <summary>A name for reports.</summary>
    public string Name { get; init; } = "a sample";

    /// <summary>The options the maker is read with (its own keys); null for its defaults.</summary>
    public IReadOnlyDictionary<string, string>? Options { get; init; }

    /// <summary>The tracer the glyphs are traced with, by name (<see cref="OutlineTracers.Contour"/> unless set).</summary>
    public string Tracer { get; init; } = OutlineTracers.Contour;

    /// <summary>The tracer's options, or null for its defaults.</summary>
    public IReadOnlyDictionary<string, string>? TracerOptions { get; init; }
}

/// <summary>
/// The checks of a font maker (<see cref="IFontMaker"/>, registered in <see cref="FontMakers"/>) on glyphs of its own
/// (<see cref="FontMakerSample"/>), each whole and as random parts:
/// <list type="bullet">
/// <item>every glyph given is made or reported: each fill of a text and form once (numbered from 0), or left out with a
/// reason; what the source left out is passed on;</item>
/// <item>every character given answers with its first fill, which has ink;</item>
/// <item>each made glyph sits where its ink sat: its top, bottom and width those of its ink (the pixels whose coverage
/// reaches half, where a tracer's edge is promised) at the size and baseline it was written at, within the tracer's
/// promise and a pixel;</item>
/// <item>each glyph that is not a mark has an advance that holds its ink;</item>
/// <item>the face's metrics are those the maker says it measured, the ascender above the baseline and the descender at or
/// below it;</item>
/// <item>the face is written as TrueType and read back with the same glyphs, advances, characters and layout tables (when a
/// "truetype" writer is registered);</item>
/// <item>the same face again and from several threads;</item>
/// <item>a request of no glyphs is refused;</item>
/// <item>an option it does not take is refused, naming it.</item>
/// </list>
/// </summary>
public sealed class FontMakerSuite : ContractSuite<IFontMaker>
{
    private readonly IReadOnlyList<FontMakerSample> _samples;

    /// <summary>The checks on <paramref name="samples"/>.</summary>
    /// <param name="samples">Glyphs in the maker's kind (at least one sample, each of at least one glyph).</param>
    /// <exception cref="ArgumentException">No sample, or a sample of no glyph.</exception>
    public FontMakerSuite(IReadOnlyList<FontMakerSample> samples)
    {
        ArgumentNullException.ThrowIfNull(samples);
        if (samples.Count == 0)
        {
            throw new ArgumentException("A font maker is checked on glyphs of its own kind: give at least one sample.", nameof(samples));
        }

        foreach (var sample in samples)
        {
            ArgumentNullException.ThrowIfNull(sample, nameof(samples));
            if (sample.Glyphs.Glyphs.Count == 0)
            {
                throw new ArgumentException($"{sample.Name}: a sample holds at least one glyph.", nameof(samples));
            }
        }

        _samples = samples;
    }

    /// <summary>The option no font maker takes, which each must refuse naming it.</summary>
    public const string UnknownOption = "font-maker-suite-unknown-option";

    /// <summary>The family name the suite gives the fonts it has made.</summary>
    public const string Family = "Suite Hand";

    /// <inheritdoc />
    public override string Name => "font-maker";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        for (int i = 0; i < _samples.Count; i++)
        {
            yield return new ContractCase(_samples[i].Name, new JsonObject { ["sample"] = i, ["part"] = 0 });
        }
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        int i = random.Next(_samples.Count);
        return new ContractCase($"{_samples[i].Name}, a random part", new JsonObject { ["sample"] = i, ["part"] = random.Next(1, int.MaxValue) });
    }

    /// <inheritdoc />
    public override void Run(IFontMaker implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var sample = _samples[(int)@case.Data["sample"]!];
        var found = Part(sample.Glyphs, (int)@case.Data["part"]!);
        CheckRefusal(implementation, checks);

        var settings = implementation.Read(sample.Options ?? PageOptions.None);
        var request = new FontMakerRequest(found, Family, settings) { Tracer = sample.Tracer, TracerOptions = sample.TracerOptions };
        var made = implementation.Make(request);
        var face = made.Face;
        int upem = face.Metrics.UnitsPerEm;
        var tracer = OutlineTracers.Get(sample.Tracer);
        float tolerance = tracer.Tolerance(tracer.Read(sample.TracerOptions ?? PageOptions.None));

        // The given glyphs by their text and form, in order: a made glyph's fill is its place among them.
        var given = new Dictionary<(string, string?), List<LabelledGlyph>>();
        foreach (var g in found.Glyphs)
        {
            var key = (g.Text, g.Form);
            if (!given.TryGetValue(key, out var fills))
            {
                given[key] = fills = [];
            }

            fills.Add(g);
        }

        checks.Compare("every glyph given made or reported", Accounted(found, made, given, face));
        checks.Compare("every character given answers with its first fill, with ink", Answered(made, given, face));
        checks.Compare("each made glyph sits where its ink sat", Placed(made, given, face, upem, tolerance));
        checks.Compare("each glyph not a mark has an advance that holds its ink", Advances(made, face));
        var m = made.Metrics;
        checks.Check("the face's metrics are those measured, the ascender above the baseline, the descender at or below it",
            face.Metrics == new FontMetrics(m.UnitsPerEm, m.Ascender, m.Descender, m.LineGap) && m.Ascender > 0 && m.Descender <= 0,
            $"the face's metrics {face.Metrics}, measured {m}");

        if (FontWriters.Names.Contains("truetype", StringComparer.OrdinalIgnoreCase) && FontFormats.Names.Count > 0)
        {
            checks.Compare("written as TrueType and read back the same", WrittenBack(made));
        }
        else
        {
            checks.Skip("written as TrueType and read back the same", "no truetype writer is registered");
        }

        string first = Fingerprint(made);
        string? again = Fingerprint(implementation.Make(request)) == first ? null : "a second make gave another face";
        var parallel = new string[4];
        Parallel.For(0, parallel.Length, k => parallel[k] = Fingerprint(implementation.Make(request)));
        checks.Compare("the same face again and from several threads", again ?? (parallel.All(p => p == first) ? null : "a make on another thread gave another face"));

        try
        {
            implementation.Make(request with { Glyphs = FoundGlyphs.None });
            checks.Fail("a request of no glyphs is refused", "a face was made of no glyphs");
        }
        catch (ArgumentException)
        {
            checks.Pass("a request of no glyphs is refused");
        }
    }

    /// <inheritdoc />
    protected override string Describe(IFontMaker implementation) => $"{implementation.Name} ({implementation.GetType().Name})";

    private static void CheckRefusal(IFontMaker implementation, CaseChecks checks)
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

    // The glyphs whole (part 0), or a random part of them (each kept at two chances in three, the first always).
    private static FoundGlyphs Part(FoundGlyphs found, int part)
    {
        if (part == 0)
        {
            return found;
        }

        var random = new Random(part);
        return found with { Glyphs = [.. found.Glyphs.Where((_, i) => i == 0 || random.Next(3) > 0)] };
    }

    // Each text and form's fills: made once each (numbered 0 on), or left out; the source's left-out passed on; every made
    // glyph a glyph of the face with ink.
    private static string? Accounted(FoundGlyphs found, MadeFont made, Dictionary<(string, string?), List<LabelledGlyph>> given, IFontFace face)
    {
        foreach (var bad in made.Glyphs.Where(g => g.Glyph <= 0 || g.Glyph >= face.GlyphCount || face.Outline(g.Glyph).IsEmpty))
        {
            return $"'{bad.Text}'{Form(bad.Form)} fill {bad.Fill} is glyph {bad.Glyph}: none of the face's {face.GlyphCount} with ink";
        }

        foreach (var ((text, form), fills) in given)
        {
            var numbers = made.Glyphs.Where(g => g.Text == text && g.Form == form).Select(g => g.Fill).Order().ToList();
            int reported = made.LeftOut.Count(l => l.Text == text && l.Form == form && !string.IsNullOrWhiteSpace(l.Reason))
                           - found.LeftOut.Count(l => l.Text == text && l.Form == form);
            if (!numbers.SequenceEqual(Enumerable.Range(0, numbers.Count)) || numbers.Count + reported != fills.Count)
            {
                return $"'{text}'{Form(form)}: {fills.Count} fills given; made as fills [{string.Join(", ", numbers)}], {reported} reported left out";
            }
        }

        var lost = found.LeftOut.FirstOrDefault(l => !made.LeftOut.Contains(l));
        return lost.Text is null ? null : $"'{lost.Text}'{Form(lost.Form)}, which the source left out ({lost.Reason}), is not passed on";
    }

    // Each character given plainly (no form, or isol) and made answers with its first fill, which has ink.
    private static string? Answered(MadeFont made, Dictionary<(string, string?), List<LabelledGlyph>> given, IFontFace face)
    {
        foreach (var glyph in made.Glyphs.Where(g => g.Fill == 0 && g.Form is null or "isol" && Single(g.Text) is not null))
        {
            int code = Single(glyph.Text)!.Value;
            int answer = face.Glyph(code);
            if (answer != glyph.Glyph || face.Outline(answer).IsEmpty)
            {
                return string.Create(CultureInfo.InvariantCulture, $"U+{code:X4} answers glyph {answer}; its first fill is glyph {glyph.Glyph}");
            }
        }

        return null;
    }

    // Each made glyph's top, bottom and width those of its ink as written (its half-ink box), within the tracer's promise and
    // a pixel (and a unit).
    private static string? Placed(MadeFont made, Dictionary<(string, string?), List<LabelledGlyph>> given, IFontFace face, int upem, float tolerance)
    {
        foreach (var glyph in made.Glyphs)
        {
            if (!given.TryGetValue((glyph.Text, glyph.Form), out var fills) || glyph.Fill >= fills.Count)
            {
                return $"'{glyph.Text}'{Form(glyph.Form)} fill {glyph.Fill} was not given";
            }

            var source = fills[glyph.Fill];
            var ink = HalfInk(source.Ink);
            float units = upem / source.Size;
            float top = (source.Baseline - (source.Box.Y + ink.Y)) * units, bottom = (source.Baseline - (source.Box.Y + ink.Bottom)) * units, width = ink.Width * units;
            float within = (tolerance + 1f) * units + 1f;
            var (minX, minY, maxX, maxY) = Extent(face.Outline(glyph.Glyph));
            if (MathF.Abs(maxY - top) > within || MathF.Abs(minY - bottom) > within || MathF.Abs(maxX - minX - width) > 2 * within)
            {
                return string.Create(CultureInfo.InvariantCulture,
                    $"'{glyph.Text}'{Form(glyph.Form)} fill {glyph.Fill}: its outline spans y {minY:0.#} to {maxY:0.#}, {maxX - minX:0.#} wide; its ink {bottom:0.#} to {top:0.#}, {width:0.#} wide (within {within:0.#} units)");
            }
        }

        return null;
    }

    // The box of the pixels whose coverage reaches half (where a tracer's edge is promised: faint fringes and specks left
    // out), or the ink's box when none does.
    private static PixelBox HalfInk(GlyphCoverage ink)
    {
        int left = ink.Width, top = -1, right = 0, bottom = 0;
        for (int y = 0; y < ink.Height; y++)
        {
            var row = ink.Row(y);
            int first = -1, last = -1;
            for (int x = 0; x < row.Length; x++)
            {
                if (row[x] >= 0.5f)
                {
                    first = first < 0 ? x : first;
                    last = x;
                }
            }

            if (first >= 0)
            {
                (left, right) = (Math.Min(left, first), Math.Max(right, last + 1));
                top = top < 0 ? y : top;
                bottom = y + 1;
            }
        }

        return top < 0 ? ink.InkBox : new PixelBox(left, top, right - left, bottom - top);
    }

    // Each made glyph that is not a mark: an advance above 0 that holds its ink (a unit's rounding either side).
    private static string? Advances(MadeFont made, IFontFace face)
    {
        foreach (var glyph in made.Glyphs)
        {
            if (Single(glyph.Text) is { } code && CharUnicodeInfo.GetUnicodeCategory(code) is UnicodeCategory.NonSpacingMark or UnicodeCategory.EnclosingMark)
            {
                continue;
            }

            int advance = face.Advance(glyph.Glyph);
            var (minX, _, maxX, _) = Extent(face.Outline(glyph.Glyph));
            if (advance <= 0 || minX < -1f || maxX > advance + 1f)
            {
                return string.Create(CultureInfo.InvariantCulture, $"'{glyph.Text}'{Form(glyph.Form)} fill {glyph.Fill}: its ink spans x {minX:0.#} to {maxX:0.#}, its advance is {advance}");
            }
        }

        return null;
    }

    // The face written as TrueType and read back: the same glyph count, advances, outlines' bounds, characters and layout tables.
    private static string? WrittenBack(MadeFont made)
    {
        var face = made.Face;
        using var stream = new MemoryStream();
        FontWriters.Write(face, stream, "truetype");
        var back = FontFormats.Read(stream.ToArray(), Family).FirstOrDefault();
        if (back is null || back.GlyphCount != face.GlyphCount)
        {
            return $"read back {back?.GlyphCount ?? 0} glyphs of {face.GlyphCount}";
        }

        for (int g = 0; g < face.GlyphCount; g++)
        {
            var (a, b) = (face.Outline(g).Bounds(), back.Outline(g).Bounds());
            if (face.Advance(g) != back.Advance(g) || MathF.Abs(a.MinX - b.MinX) > 1f || MathF.Abs(a.MinY - b.MinY) > 1f || MathF.Abs(a.MaxX - b.MaxX) > 1f || MathF.Abs(a.MaxY - b.MaxY) > 1f)
            {
                return $"glyph {g}: advance {back.Advance(g)} and bounds {b} read back, {face.Advance(g)} and {a} written";
            }
        }

        foreach (var glyph in made.Glyphs.Where(g => g.Fill == 0 && Single(g.Text) is not null))
        {
            int code = Single(glyph.Text)!.Value;
            if (face.Glyph(code) != back.Glyph(code))
            {
                return string.Create(CultureInfo.InvariantCulture, $"U+{code:X4}: glyph {back.Glyph(code)} read back, {face.Glyph(code)} written");
            }
        }

        foreach (string tag in (string[])["GSUB", "GDEF", "GPOS"])
        {
            if (!face.Table(tag).Span.SequenceEqual(back.Table(tag).Span))
            {
                return $"the {tag} table read back is not the one written ({back.Table(tag).Length} bytes, {face.Table(tag).Length} written)";
            }
        }

        return null;
    }

    // A face as a hash: its metrics, every glyph's steps, points and advance, its layout tables and the glyphs made.
    private static string Fingerprint(MadeFont made)
    {
        var face = made.Face;
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var text = new StringBuilder();
        text.Append(face.Metrics).Append('|').Append(face.GlyphCount).Append('|').Append(made.Metrics);
        foreach (var g in made.Glyphs)
        {
            text.Append('|').Append(g);
        }

        hash.AppendData(Encoding.UTF8.GetBytes(text.ToString()));
        Span<byte> number = stackalloc byte[4];
        for (int g = 0; g < face.GlyphCount; g++)
        {
            var outline = face.Outline(g);
            foreach (var verb in outline.Verbs.Span)
            {
                number[0] = (byte)verb;
                hash.AppendData(number[..1]);
            }

            foreach (float p in outline.Points.Span)
            {
                BitConverter.TryWriteBytes(number, p);
                hash.AppendData(number);
            }

            BitConverter.TryWriteBytes(number, face.Advance(g));
            hash.AppendData(number);
        }

        foreach (string tag in (string[])["GSUB", "GDEF", "GPOS"])
        {
            hash.AppendData(face.Table(tag).Span);
        }

        return Convert.ToHexString(hash.GetHashAndReset());
    }

    // The outline's true extent: its points on the outline and its curves' turning points (a control point off the outline
    // is not ink).
    private static (float MinX, float MinY, float MaxX, float MaxY) Extent(GlyphOutline outline)
    {
        var verbs = outline.Verbs.Span;
        var p = outline.Points.Span;
        float minX = float.PositiveInfinity, minY = float.PositiveInfinity, maxX = float.NegativeInfinity, maxY = float.NegativeInfinity;
        void Add(double x, double y)
        {
            (minX, maxX) = (MathF.Min(minX, (float)x), MathF.Max(maxX, (float)x));
            (minY, maxY) = (MathF.Min(minY, (float)y), MathF.Max(maxY, (float)y));
        }

        double cx = 0, cy = 0;
        int at = 0;
        foreach (var verb in verbs)
        {
            switch (verb)
            {
                case OutlineVerb.Move or OutlineVerb.Line:
                    (cx, cy) = (p[at], p[at + 1]);
                    Add(cx, cy);
                    at += 2;
                    break;
                case OutlineVerb.Quadratic:
                {
                    double qx = p[at], qy = p[at + 1], ex = p[at + 2], ey = p[at + 3];
                    foreach (double t in (double[])[Turn(cx, qx, ex), Turn(cy, qy, ey)])
                    {
                        if (t is > 0 and < 1)
                        {
                            double u = 1 - t;
                            Add(u * u * cx + 2 * u * t * qx + t * t * ex, u * u * cy + 2 * u * t * qy + t * t * ey);
                        }
                    }

                    (cx, cy) = (ex, ey);
                    Add(cx, cy);
                    at += 4;
                    break;
                }

                case OutlineVerb.Cubic:
                {
                    double ax = p[at], ay = p[at + 1], bx = p[at + 2], by = p[at + 3], ex = p[at + 4], ey = p[at + 5];
                    for (int k = 1; k < 64; k++)
                    {
                        double t = k / 64.0, u = 1 - t;
                        Add(u * u * u * cx + 3 * u * u * t * ax + 3 * u * t * t * bx + t * t * t * ex, u * u * u * cy + 3 * u * u * t * ay + 3 * u * t * t * by + t * t * t * ey);
                    }

                    (cx, cy) = (ex, ey);
                    Add(cx, cy);
                    at += 6;
                    break;
                }
            }
        }

        return float.IsFinite(minX) ? (minX, minY, maxX, maxY) : default;
    }

    // Where a quadratic curve from a over b to c turns on one axis (its parameter), or -1 when it does not.
    private static double Turn(double a, double b, double c)
    {
        double d = a - 2 * b + c;
        return Math.Abs(d) < 1e-12 ? -1 : (a - b) / d;
    }

    // The code point of a text of one character, or null.
    private static int? Single(string text)
    {
        var runes = text.EnumerateRunes();
        if (!runes.MoveNext())
        {
            return null;
        }

        int code = runes.Current.Value;
        return runes.MoveNext() ? null : code;
    }

    private static string Form(string? form) => form is null ? "" : $" ({form})";
}
