// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.PluginTests;

/// <summary>
/// An outline tracer of one's own, written on Idrak's public API alone (plan 38, F0): each row's runs of pixels with at
/// least <see cref="LevelKey"/> of ink (0.5 unless set) become rectangles of whole pixels, all turning the same way, so
/// the outline has one winding and keeps its holes; its edges lie on pixel edges, within a pixel of the half-ink edge.
/// </summary>
/// <param name="name">The name it registers under.</param>
/// <param name="behaviour">How it misbehaves (the guards' proof): it fails, or its outline lies four pixels to the right.</param>
public sealed class ToyOutlineTracer(string name = ToyOutlineTracer.DefaultName, ToyBehaviour behaviour = ToyBehaviour.Plain) : IOutlineTracer
{
    /// <summary>The toy's name.</summary>
    public const string DefaultName = "outside-runs";

    /// <summary>The option that sets the ink a pixel needs to be traced, from 0.01 to 1 (0.5 unless set).</summary>
    public const string LevelKey = "toy-level";

    /// <inheritdoc />
    public string Name { get; } = name;

    /// <inheritdoc />
    public string Summary => "each row's runs of ink as rectangles of whole pixels (the outside proof's tracer)";

    /// <inheritdoc />
    public IReadOnlySet<string> Keys { get; } = new OptionKeys(LevelKey);

    /// <inheritdoc />
    public OutlineTracerSettings Read(IReadOnlyDictionary<string, string> options)
    {
        ArgumentNullException.ThrowIfNull(options);
        float level = 0.5f;
        foreach (var (key, value) in options)
        {
            if (!Keys.Contains(key.ToLowerInvariant()))
            {
                throw new ArgumentException($"The tracer {Name} takes {LevelKey} only, not '{key}'.", nameof(options));
            }

            level = PageOptions.Number($"The tracer {Name}'s {LevelKey}", value, 0.01f, 1f);
        }

        return new Settings(level, options);
    }

    /// <inheritdoc />
    public float Tolerance(OutlineTracerSettings settings) => 1f;

    /// <inheritdoc />
    public GlyphOutline Trace(GlyphCoverage coverage, OutlinePlacement placement, OutlineTracerSettings settings)
    {
        ArgumentNullException.ThrowIfNull(coverage);
        if (!placement.IsValid)
        {
            throw new ArgumentException($"Not a placement: {placement}.", nameof(placement));
        }

        if (behaviour == ToyBehaviour.Fails)
        {
            throw new InvalidOperationException("The outside toy tracer fails, on purpose.");
        }

        float level = settings is Settings own ? own.Level : 0.5f, shift = behaviour == ToyBehaviour.Differs ? 4f : 0f;
        var verbs = new List<OutlineVerb>();
        var points = new List<float>();
        for (int y = 0; y < coverage.Height; y++)
        {
            var row = coverage.Row(y);
            for (int x = 0; x < row.Length;)
            {
                if (row[x] < level)
                {
                    x++;
                    continue;
                }

                int start = x;
                while (x < row.Length && row[x] >= level)
                {
                    x++;
                }

                // Clockwise with y up: the top left, the top right, the bottom right, the bottom left.
                var (left, top) = placement.ToUnits(start + shift, y);
                var (right, bottom) = placement.ToUnits(x + shift, y + 1);
                verbs.AddRange([OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close]);
                points.AddRange([left, top, right, top, right, bottom, left, bottom]);
            }
        }

        return verbs.Count == 0 ? GlyphOutline.Empty : new GlyphOutline([.. verbs], [.. points]);
    }

    private sealed class Settings(float level, IReadOnlyDictionary<string, string> options) : OutlineTracerSettings(options)
    {
        public float Level => level;
    }
}

/// <summary>
/// A glyph source of one's own, written on Idrak's public API alone (plan 38, F0): the ink (one less the grey, the
/// channels' mean) between blank columns is one glyph each, its region the band of columns from the image's top to its
/// bottom, labelled with the known text's characters (spaces aside) from the left; a character with no ink left for it
/// is reported left out. A column holds ink where a pixel has <see cref="LevelKey"/> of it (0.5 unless set). The baseline
/// is the lowest ink's bottom row, the size the tallest glyph's height.
/// </summary>
/// <param name="name">The name it registers under.</param>
/// <param name="behaviour">How it misbehaves (the guards' proof): it fails, or it labels the glyphs from the right.</param>
public sealed class ToyGlyphSource(string name = ToyGlyphSource.DefaultName, ToyBehaviour behaviour = ToyBehaviour.Plain) : IGlyphSource
{
    /// <summary>The toy's name.</summary>
    public const string DefaultName = "outside-columns";

    /// <summary>The option that sets the ink a pixel needs to count, from 0.01 to 1 (0.5 unless set).</summary>
    public const string LevelKey = "toy-ink";

    /// <inheritdoc />
    public string Name { get; } = name;

    /// <inheritdoc />
    public string Summary => "the ink between blank columns, one glyph each, labelled from the known text (the outside proof's source)";

    /// <inheritdoc />
    public IReadOnlySet<string> Keys { get; } = new OptionKeys(LevelKey);

    /// <inheritdoc />
    public GlyphSourceSettings Read(IReadOnlyDictionary<string, string> options)
    {
        ArgumentNullException.ThrowIfNull(options);
        float level = 0.5f;
        foreach (var (key, value) in options)
        {
            if (!Keys.Contains(key.ToLowerInvariant()))
            {
                throw new ArgumentException($"The glyph source {Name} takes {LevelKey} only, not '{key}'.", nameof(options));
            }

            level = PageOptions.Number($"The glyph source {Name}'s {LevelKey}", value, 0.01f, 1f);
        }

        return new Settings(level, options);
    }

    /// <inheritdoc />
    public FoundGlyphs Find(GlyphSourceRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (behaviour == ToyBehaviour.Fails)
        {
            throw new InvalidOperationException("The outside toy glyph source fails, on purpose.");
        }

        float level = request.Settings is Settings own ? own.Level : 0.5f;
        var image = request.Image;
        int width = image.Width, height = image.Height, size = width * height, channels = image.Channels;

        // The ink, one plane, the channels walked in memory order; and the columns holding ink.
        var pixels = image.Pixels;
        var ink = new float[size];
        for (int c = 0; c < channels; c++)
        {
            var plane = pixels.AsSpan(c * size, size);
            for (int i = 0; i < size; i++)
            {
                ink[i] += plane[i];
            }
        }

        var inked = new bool[width];
        for (int i = 0; i < size; i++)
        {
            ink[i] = Math.Clamp(1f - ink[i] / channels, 0f, 1f);
            inked[i % width] |= ink[i] >= level;
        }

        // Bands of inked columns, each one glyph's region; its box, the rows holding ink.
        var coverage = new GlyphCoverage(ink, width, height);
        var boxes = new List<(PixelBox Box, PixelBox Region)>();
        for (int x = 0; x < width;)
        {
            if (!inked[x])
            {
                x++;
                continue;
            }

            int start = x;
            while (x < width && inked[x])
            {
                x++;
            }

            int top = -1, bottom = -1;
            for (int y = 0; y < height; y++)
            {
                if (RowHolds(ink, y * width + start, x - start, level))
                {
                    top = top < 0 ? y : top;
                    bottom = y + 1;
                }
            }

            boxes.Add((new PixelBox(start, top, x - start, bottom - top), new PixelBox(start, 0, x - start, height)));
        }

        var labels = Labels(request.Text);
        if (behaviour == ToyBehaviour.Differs)
        {
            labels.Reverse();
        }

        float baseline = boxes.Count == 0 ? 0 : boxes.Max(b => b.Box.Bottom), em = boxes.Count == 0 ? 1 : boxes.Max(b => b.Box.Height);
        var glyphs = new List<LabelledGlyph>();
        var leftOut = new List<GlyphLeftOut>();
        for (int i = 0; i < labels.Count; i++)
        {
            if (i < boxes.Count)
            {
                glyphs.Add(new LabelledGlyph(labels[i], null, coverage.Crop(boxes[i].Box), boxes[i].Box, boxes[i].Region, baseline, em));
            }
            else
            {
                leftOut.Add(new GlyphLeftOut(labels[i], null, "no ink left for it"));
            }
        }

        return new FoundGlyphs(glyphs, leftOut);
    }

    // Whether the row's part holds a pixel with level of ink.
    private static bool RowHolds(float[] ink, int start, int length, float level)
    {
        foreach (float v in ink.AsSpan(start, length))
        {
            if (v >= level)
            {
                return true;
            }
        }

        return false;
    }

    // The text's characters (text elements), spaces aside, in order.
    private static List<string> Labels(string? text)
    {
        var labels = new List<string>();
        var elements = StringInfo.GetTextElementEnumerator(text ?? "");
        while (elements.MoveNext())
        {
            string element = elements.GetTextElement();
            if (!string.IsNullOrWhiteSpace(element))
            {
                labels.Add(element);
            }
        }

        return labels;
    }

    private sealed class Settings(float level, IReadOnlyDictionary<string, string> options) : GlyphSourceSettings(options)
    {
        public float Level => level;
    }
}

/// <summary>
/// The proofs of the handwriting contracts from outside (plan 38, F0): <see cref="ToyOutlineTracer"/> and
/// <see cref="ToyGlyphSource"/> registered, used by name, checked by the conformance kit, and put over a library tracer
/// and source of the same name to fall back and be compared.
/// </summary>
public static class HandwritingPluginTests
{
    /// <summary>The name the proof's library defaults and app registrations share.</summary>
    public const string ProofName = "outside-proof";

    private static readonly Lock Gate = new();
    private static bool s_defaults;

    /// <summary>
    /// Pages of generic glyph shapes in a row (a bar, a ring, a hook, a block with a notch), with their known text and
    /// where each glyph is: a grey page, a colour page read at another ink level, and a page whose text has a character more
    /// than it has ink (reported left out).
    /// </summary>
    public static IReadOnlyList<Abstraction.Testing.GlyphSourceSample> Samples()
    {
        var (grey, greyBoxes) = Page("1074", channels: 1);
        var (colour, colourBoxes) = Page("4701", channels: 3);
        var (short1, shortBoxes) = Page("170", channels: 1);
        return
        [
            new(grey, "10 74") { Name = "a grey row of four", Glyphs = Expected("1074", greyBoxes) },
            new(colour, "4701") { Name = "a colour row of four", Glyphs = Expected("4701", colourBoxes), Options = new Dictionary<string, string> { [ToyGlyphSource.LevelKey] = "0.4" } },
            new(short1, "1 7 0 4") { Name = "a row with a character more than its ink", Glyphs = Expected("170", shortBoxes) },
        ];
    }

    /// <summary>
    /// A toy tracer and source of one's own register, are found by name, find and trace a page's glyphs, pass the suites
    /// and unregister; an unknown name is refused naming the registered ones.
    /// </summary>
    public static void RegistersAndConforms(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;                                                          // read on the host: checked once
        }

        var tracer = new ToyOutlineTracer();
        var source = new ToyGlyphSource();
        OutlineTracers.Register(tracer);
        GlyphSources.Register(source);
        try
        {
            string origin = typeof(HandwritingPluginTests).Assembly.GetName().Name!;
            Check(OutlineTracers.Names.Contains(ToyOutlineTracer.DefaultName) && OutlineTracers.Origin(ToyOutlineTracer.DefaultName) == origin
                  && OutlineTracers.Default(ToyOutlineTracer.DefaultName) is null && ReferenceEquals(OutlineTracers.Get("OUTSIDE-RUNS"), tracer)
                  && OutlineTracers.Describe().Contains($"{ToyOutlineTracer.DefaultName}: ", StringComparison.Ordinal) && OutlineTracers.AllKeys().Contains(ToyOutlineTracer.LevelKey),
                $"the tracer registered from outside: {string.Join(", ", OutlineTracers.Names)}, origin {OutlineTracers.Origin(ToyOutlineTracer.DefaultName)}");
            Check(GlyphSources.Names.Contains(ToyGlyphSource.DefaultName) && GlyphSources.Origin(ToyGlyphSource.DefaultName) == origin
                  && GlyphSources.Default(ToyGlyphSource.DefaultName) is null && ReferenceEquals(GlyphSources.Find("Outside-Columns"), source)
                  && GlyphSources.Describe().Contains($"{ToyGlyphSource.DefaultName}: ", StringComparison.Ordinal) && GlyphSources.AllKeys().Contains(ToyGlyphSource.LevelKey),
                $"the source registered from outside: {string.Join(", ", GlyphSources.Names)}");

            // Used by name: a page's glyphs found, labelled from its text, and traced at their baseline and size.
            var sample = Samples()[0];
            var found = GlyphSources.Get(ToyGlyphSource.DefaultName).Find(new GlyphSourceRequest(sample.Image, sample.Text, source.Read(PageOptions.None)));
            Check(string.Concat(found.Glyphs.Select(g => g.Text)) == "1074" && found.LeftOut.Count == 0, $"found: {string.Join("; ", found.Glyphs)}");
            var settings = tracer.Read(PageOptions.None);
            foreach (var glyph in found.Glyphs)
            {
                var outline = OutlineTracers.Get(ToyOutlineTracer.DefaultName).Trace(glyph.Ink, glyph.Placement(1000), settings);
                var (minX, minY, maxX, maxY) = outline.Bounds();
                float units = 1000f / glyph.Size;
                Check(!outline.IsEmpty && MathF.Abs(minX) < 0.5f && MathF.Abs(maxX - glyph.Box.Width * units) < 1f
                      && MathF.Abs(maxY - (glyph.Baseline - glyph.Box.Y) * units) < 1f && minY >= -1f,
                    $"'{glyph.Text}' traced at its baseline and size: {outline.Bounds()} for {glyph}");
            }

            // The conformance kit on the toys.
            Abstraction.Testing.Conformance.CheckOutlineTracer(tracer, settings: [new Dictionary<string, string> { [ToyOutlineTracer.LevelKey] = "0.4" }]).ThrowIfFailed();
            Abstraction.Testing.Conformance.CheckGlyphSource(source, Samples(), new Abstraction.Testing.ContractCheckOptions { RandomCases = 8 }).ThrowIfFailed();
        }
        finally
        {
            Check(OutlineTracers.Unregister(ToyOutlineTracer.DefaultName) && GlyphSources.Unregister(ToyGlyphSource.DefaultName)
                  && OutlineTracers.Find(ToyOutlineTracer.DefaultName) is null && GlyphSources.Find(ToyGlyphSource.DefaultName) is null, "unregistered");
        }

        try
        {
            OutlineTracers.Get(ToyOutlineTracer.DefaultName);
            Check(false, "an unregistered tracer was found");
        }
        catch (NotRegisteredException e)
        {
            Check(e.Registry == nameof(OutlineTracers) && e.Name == ToyOutlineTracer.DefaultName && e.Registered.SequenceEqual(OutlineTracers.Names)
                  && e.Message.Contains("OutlineTracers.Register", StringComparison.Ordinal), $"the refusal: {e.Message}");
        }

        try
        {
            GlyphSources.Get(ToyGlyphSource.DefaultName);
            Check(false, "an unregistered source was found");
        }
        catch (NotRegisteredException e)
        {
            Check(e.Registry == nameof(GlyphSources) && e.Name == ToyGlyphSource.DefaultName && e.Registered.SequenceEqual(GlyphSources.Names)
                  && e.Message.Contains("GlyphSources.Register", StringComparison.Ordinal), $"the refusal: {e.Message}");
        }
    }

    /// <summary>
    /// A toy tracer and source over library ones of the same name (stand-ins for the library's, registered as its
    /// defaults for the process): the app's answers by default and its failure reaches the caller; under FallBack the
    /// library's answers the failing call; under Shadow the library's answers and the app's is compared; unregistered,
    /// the library's are back.
    /// </summary>
    public static void OverridesFallsBackAndShadows(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        lock (Gate)
        {
            if (!s_defaults)
            {
                // Library defaults stay for the process: the stand-ins for a tracer and a source the library ships.
                Overrides.AsLibraryDefaults(() =>
                {
                    OutlineTracers.Register(new ToyOutlineTracer(ProofName));
                    GlyphSources.Register(new ToyGlyphSource(ProofName));
                });
                s_defaults = true;
            }
        }

        Check(OutlineTracers.Origin(ProofName) == Overrides.Library && GlyphSources.Origin(ProofName) == Overrides.Library, "the stand-ins are library defaults");
        var sample = Samples()[0];
        var library = GlyphSources.Get(ProofName);
        var expected = library.Find(new GlyphSourceRequest(sample.Image, sample.Text, library.Read(PageOptions.None)));
        var glyph = expected.Glyphs[1];
        var at = glyph.Placement(1000);
        var libraryTracer = OutlineTracers.Get(ProofName);
        var traced = libraryTracer.Trace(glyph.Ink, at, libraryTracer.Read(PageOptions.None));
        string app = typeof(HandwritingPluginTests).Assembly.GetName().Name!;
        try
        {
            // The tracer: fails, then traces four pixels off.
            OutlineTracers.Register(new ToyOutlineTracer(ProofName, ToyBehaviour.Fails));
            Check(OutlineTracers.Origin(ProofName) == app && OutlineTracers.Default(ProofName) is not null, "the app's tracer over the library's");
            var guarded = OutlineTracers.Get(ProofName);
            var settings = guarded.Read(PageOptions.None);
            try
            {
                guarded.Trace(glyph.Ink, at, settings);
                Check(false, "the app's tracer's failure was not reported");
            }
            catch (InvalidOperationException)
            {
                // Throw, the default policy: the app's failure reaches the caller.
            }

            OutlineTracers.SetPolicy(ProofName, SlotPolicy.FallBack);
            Check(Same(guarded.Trace(glyph.Ink, at, settings), traced), "under FallBack the library's tracer answers the failing call");
            Check(Row(nameof(OutlineTracers)) is { Failures: >= 2, FallBacks: >= 1 }, $"the report: {Row(nameof(OutlineTracers))}");
            OutlineTracers.Register(new ToyOutlineTracer(ProofName, ToyBehaviour.Differs));
            OutlineTracers.SetPolicy(ProofName, SlotPolicy.Shadow, shadowRate: 1);
            var shadowed = OutlineTracers.Get(ProofName);
            Check(Same(shadowed.Trace(glyph.Ink, at, shadowed.Read(PageOptions.None)), traced), "under Shadow the library's tracer answers");
            Check(Row(nameof(OutlineTracers)) is { Compared: >= 1, Differed: >= 1 }, $"the tracers' comparison: {Row(nameof(OutlineTracers))}");

            // The source: fails, then labels from the right.
            GlyphSources.Register(new ToyGlyphSource(ProofName, ToyBehaviour.Fails));
            var source = GlyphSources.Get(ProofName);
            var request = new GlyphSourceRequest(sample.Image, sample.Text, source.Read(PageOptions.None));
            try
            {
                source.Find(request);
                Check(false, "the app's source's failure was not reported");
            }
            catch (InvalidOperationException)
            {
                // Throw, the default policy.
            }

            GlyphSources.SetPolicy(ProofName, SlotPolicy.FallBack);
            Check(Labels(source.Find(request)) == Labels(expected), "under FallBack the library's source answers the failing call");
            Check(Row(nameof(GlyphSources)) is { Failures: >= 2, FallBacks: >= 1 }, $"the report: {Row(nameof(GlyphSources))}");
            GlyphSources.Register(new ToyGlyphSource(ProofName, ToyBehaviour.Differs));
            GlyphSources.SetPolicy(ProofName, SlotPolicy.Shadow, shadowRate: 1);
            var compared = GlyphSources.Get(ProofName);
            Check(Labels(compared.Find(request with { Settings = compared.Read(PageOptions.None) })) == Labels(expected), "under Shadow the library's source answers");
            Check(Row(nameof(GlyphSources)) is { Compared: >= 1, Differed: >= 1 }, $"the sources' comparison: {Row(nameof(GlyphSources))}");
        }
        finally
        {
            OutlineTracers.Unregister(ProofName);
            GlyphSources.Unregister(ProofName);
            OutlineTracers.SetPolicy(ProofName, SlotPolicy.Throw);
            GlyphSources.SetPolicy(ProofName, SlotPolicy.Throw);
        }

        Check(OutlineTracers.Origin(ProofName) == Overrides.Library && GlyphSources.Origin(ProofName) == Overrides.Library, "unregistered, the library's are back");
    }

    private static SlotOverride? Row(string registry) => Overrides.Report().FirstOrDefault(o => o.Registry == registry && o.Name == ProofName);

    private static bool Same(GlyphOutline a, GlyphOutline b) => a.Verbs.Span.SequenceEqual(b.Verbs.Span) && a.Points.Span.SequenceEqual(b.Points.Span);

    private static string Labels(FoundGlyphs found) => string.Join(" ", found.Glyphs.Select(g => $"{g.Text}@{g.Box}"));

    private static List<Abstraction.Testing.ExpectedGlyph> Expected(string text, IReadOnlyList<PixelBox> boxes) =>
        [.. text.Select((c, i) => new Abstraction.Testing.ExpectedGlyph(c.ToString(), null, boxes[i]))];

    // A page of the glyphs of text (digits drawn as generic shapes), 14 pixels wide and 24 tall, 6 apart, on white paper;
    // and each glyph's box.
    private static (ImageData Image, IReadOnlyList<PixelBox> Boxes) Page(string text, int channels)
    {
        const int Width = 14, Height = 24, Gap = 6, Top = 8, Left = 5;
        int width = Left * 2 + text.Length * Width + (text.Length - 1) * Gap, height = Top * 2 + Height;
        var plane = new float[width * height];
        Array.Fill(plane, 1f);
        var boxes = new List<PixelBox>();
        for (int i = 0; i < text.Length; i++)
        {
            int x = Left + i * (Width + Gap);
            void Ink(int dx, int dy, int w, int h)
            {
                for (int y = Top + dy; y < Top + dy + h; y++)
                {
                    plane.AsSpan(y * width + x + dx, w).Fill(0.05f);
                }
            }

            switch (text[i])
            {
                case '1':
                    Ink(5, 0, 4, Height);                                        // a bar
                    boxes.Add(new PixelBox(x + 5, Top, 4, Height));
                    break;
                case '0':
                    Ink(0, 0, Width, 4);                                         // a ring
                    Ink(0, Height - 4, Width, 4);
                    Ink(0, 0, 4, Height);
                    Ink(Width - 4, 0, 4, Height);
                    boxes.Add(new PixelBox(x, Top, Width, Height));
                    break;
                case '7':
                    Ink(0, 0, Width, 4);                                         // a hook
                    Ink(Width - 4, 0, 4, Height);
                    boxes.Add(new PixelBox(x, Top, Width, Height));
                    break;
                default:
                    Ink(0, 0, 4, Height / 2 + 2);                                // a block with a notch
                    Ink(0, Height / 2 - 2, Width, 4);
                    Ink(Width - 4, 0, 4, Height);
                    boxes.Add(new PixelBox(x, Top, Width, Height));
                    break;
            }
        }

        var pixels = new float[channels * plane.Length];
        for (int c = 0; c < channels; c++)
        {
            plane.CopyTo(pixels, c * plane.Length);
        }

        return (new ImageData(pixels, channels, height, width), boxes);
    }

    private static void Check(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }
}
