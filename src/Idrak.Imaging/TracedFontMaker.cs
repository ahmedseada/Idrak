// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Globalization;
using System.Text;
using System.Text.Json;

namespace Idrak.Imaging;

/// <summary>
/// "traced" (plan 38, F3): labelled glyphs made into a font. Each glyph's ink is traced into its outline by the request's
/// tracer at the baseline and size it was written at (<see cref="LabelledGlyph.Placement"/>), the glyphs traced in
/// parallel (<see cref="ThreadsKey"/>). The face's metrics are measured from the ink of the glyphs that are not marks,
/// read as the font maps read a font's ink (<see cref="InkMeasure"/>: 256 pixels per em): the ascender and the descender
/// hold <see cref="InkShareKey"/> of the glyphs' ink tops and bottoms (the others are outliers: a stray stroke does not
/// open every line), the x-height and the cap height are the median tops of the short and the tall glyphs among those at
/// least half the median ink height (a dot, a comma or a dash is neither), split by Otsu's level of their tops, and the
/// pen is the strokes' median thickness (<see cref="Heights"/>). Each glyph is placed with a side bearing either
/// side of its ink (<see cref="SideBearingKey"/>: half the pen unless set), none for the forms that join their neighbours
/// (<see cref="JoinedFormsKey"/>) and for marks; its advance reaches the bearing past its ink. The outlines are kept as
/// TrueType gives them back (whole units; <see cref="GlyphEncoder.Whole"/>), so the font written from the face is the face.
/// The forms, ligatures, marks (placed above or below from where their ink sat) and alternates (a character's other fills)
/// are written into a font map as data (glyphs by "#id"; the script tags and the ligatures' feature from the settings),
/// which <c>FontMaps.Apply</c> applies: the "opentype" mapper makes the GSUB, GDEF and GPOS
/// tables, anchors the marks from the ink and draws the digits and marks the glyphs lack, sized from the written glyphs
/// (<see cref="MeasureKey"/>). Nothing names a script: the characters, forms and tags are the labels and the settings.
/// The maker holds no state; fonts are made from several threads at once.
/// </summary>
internal sealed class TracedFontMaker : IFontMaker
{
    /// <summary>The face's units per em (16 to 16,384; 1,000 unless set).</summary>
    internal const string UnitsKey = "units-per-em";

    /// <summary>The space either side of a glyph's ink: "auto" (half the pen measured from the ink; the default) or a share of the em from 0 to 0.5.</summary>
    internal const string SideBearingKey = "side-bearing";

    /// <summary>The joining forms whose glyphs take no side bearing, their strokes meeting their neighbours' (a list between ':'; init:medi:fina unless set; none for no form).</summary>
    internal const string JoinedFormsKey = "joined-forms";

    /// <summary>The share of the glyphs' ink tops (and bottoms) the ascender (and descender) hold, the rest outliers (0.5 to 1; 0.95 unless set).</summary>
    internal const string InkShareKey = "ink-share";

    /// <summary>The gap between lines, a share of the em (0 to 2; 0.2 unless set).</summary>
    internal const string LineGapKey = "line-gap";

    /// <summary>The space character's advance: "auto" (half the median advance of the written glyphs; the default) or a share of the em from 0 to 2.</summary>
    internal const string SpaceKey = "space";

    /// <summary>The OpenType script tags the layout tables list (a list between ':'; DFLT, the default script every shaper falls back to, unless set).</summary>
    internal const string ScriptsKey = "scripts";

    /// <summary>The OpenType feature the ligatures are made under (liga unless set).</summary>
    internal const string LigatureFeatureKey = "ligature-feature";

    /// <summary>Whether a character's further fills are kept as its alternates (on) or left out (off).</summary>
    internal const string AlternatesKey = "alternates";

    /// <summary>The characters whose glyphs size the drawn digits (a list between ':' of U+XXXX or single characters; none unless set: the written letters).</summary>
    internal const string MeasureKey = "measure";

    /// <summary>The gap between a base's ink and the mark placed on it, a share of the em (0 to 1; 0.06 unless set).</summary>
    internal const string MarkGapKey = "mark-gap";

    /// <summary>Half the width around a base's middle where its ink top and bottom are read for its marks, a share of the em (0.01 to 2; 0.13 unless set).</summary>
    internal const string MarkWindowKey = "mark-window";

    /// <summary>The threads glyphs are traced on: "auto" (<see cref="ComputeResources.MaxCpuThreads"/>; the default) or 1 to 1,024 (1: one thread).</summary>
    internal const string ThreadsKey = "threads";

    // The forms a ligature's components take inside a whole of each form: the first, the middle ones, the last.
    private static readonly Dictionary<string, (string First, string Middle, string Last)> LigatureForms = new(StringComparer.Ordinal)
    {
        ["isol"] = ("init", "medi", "fina"),
        ["init"] = ("init", "medi", "medi"),
        ["medi"] = ("medi", "medi", "medi"),
        ["fina"] = ("medi", "medi", "fina"),
    };

    private static readonly string Own = string.Join(", ", UnitsKey, SideBearingKey, JoinedFormsKey, InkShareKey, LineGapKey, SpaceKey, ScriptsKey, LigatureFeatureKey, AlternatesKey,
        MeasureKey, MarkGapKey, MarkWindowKey, ThreadsKey);

    private static readonly Settings Defaults = new(PageOptions.None);

    /// <inheritdoc/>
    public string Name => FontMakers.Traced;

    /// <inheritdoc/>
    public string Summary => "each glyph traced at its baseline and size; ascender, descender, x-height and pen measured from the ink; side bearings from the pen; forms, ligatures, "
                             + "marks and a character's other fills (alternates) made into a font map's GSUB, GDEF and GPOS; the digits and marks the glyphs lack drawn";

    /// <inheritdoc/>
    /// <inheritdoc/>
    public FontMakerSettings Read(IReadOnlyDictionary<string, string> options)
    {
        ArgumentNullException.ThrowIfNull(options);
        return options.Count == 0 ? Defaults : new Settings(options);
    }

    // Picking the outline tracer by name, and the mapper the font maps name, is not on the new design yet. The defaults are reached directly: the
    // "contour" tracer (FontMakerRequest.Tracer's default) and the "opentype" mapper (FontMaps.Mapper's default); another
    // name stops.
    private static readonly ContourTracer Contour = new();

    private static readonly OpenTypeFontMapper OpenType = new();

    private static IOutlineTracer TracerNamed(string name) =>
        name.Equals(OutlineTracers.Contour, StringComparison.OrdinalIgnoreCase) ? Contour
        : throw new NotSupportedException($"The outline tracer '{name}' by name is not on the new design yet.");

    // The face the map makes of the made face, given for it (no detection), as FontMaps applied a map: the glyphs it lacks
    // drawn as FontMaps.Drawing says.
    private static IFontFace Apply(IFontFace face, FontMap map) =>
        !FontMaps.Mapper.Equals(FontMappers.OpenType, StringComparison.OrdinalIgnoreCase)
            ? throw new NotSupportedException($"{map.Source}: applying a font map through the font mapper '{FontMaps.Mapper}' by name is not on the new design yet.")
            : OpenType.Map(face, map, detect: false, FontMaps.Drawing)
              ?? throw new InvalidDataException($"{map.Source}: none of the map's {map.Letters.Count} letters is in the font {face.Family} {face.Style}.");

    /// <inheritdoc/>
    public MadeFont Make(FontMakerRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        var own = request.Settings as Settings ?? (Settings)Read(request.Settings.Options);
        var found = request.Glyphs.Glyphs;
        if (found.Count == 0)
        {
            throw new ArgumentException("A font is made from glyphs; the request has none (its source found no ink, or left every character out).", nameof(request));
        }

        int upem = own.UnitsPerEm;
        double window = own.MarkWindow * upem;
        var tracer = TracerNamed(request.Tracer);
        var tracerSettings = tracer.Read(request.TracerOptions ?? PageOptions.None);

        // Every glyph traced and its ink measured, independent of the others: on several threads (rule 79).
        var outlines = new GlyphOutline[found.Count];
        var inks = new InkStats?[found.Count];
        void Trace(int i)
        {
            var glyph = found[i];
            outlines[i] = tracer.Trace(glyph.Ink, glyph.Placement(upem), tracerSettings);
            inks[i] = InkMeasure.Of(outlines[i], upem, window);
        }

        int threads = own.Threads > 0 ? own.Threads : ComputeResources.MaxCpuThreads;
        if (threads > 1 && found.Count > 1)
        {
            Parallel.For(0, found.Count, new ParallelOptions { MaxDegreeOfParallelism = threads }, Trace);
        }
        else
        {
            for (int i = 0; i < found.Count; i++)
            {
                Trace(i);
            }
        }

        // Each glyph's role: its text composed, its codes, its form (a single character's isol is its plain glyph).
        var roles = new Role[found.Count];
        for (int i = 0; i < found.Count; i++)
        {
            string text = found[i].Text.IsNormalized(NormalizationForm.FormC) ? found[i].Text : found[i].Text.Normalize(NormalizationForm.FormC);
            int[] codes = [.. text.EnumerateRunes().Select(r => r.Value)];
            string? form = codes.Length == 1 && found[i].Form == "isol" ? null : found[i].Form;
            roles[i] = new Role(text, codes, form, codes.Length == 1 && UnicodeText.IsMark(codes[0]));
        }

        var metrics = MeasureInk(own, roles, inks);
        var leftOut = new List<GlyphLeftOut>(request.Glyphs.LeftOut);

        // Each glyph placed: a side bearing either side of its ink (none for joined forms and marks), kept as TrueType
        // gives it back; its advance reaches the bearing past its ink.
        double tolerance = new FontWriteOptions().CurveTolerance;
        var placed = new (GlyphOutline Outline, int Advance)[found.Count];
        for (int i = 0; i < found.Count; i++)
        {
            if (outlines[i].IsEmpty)
            {
                continue;
            }

            int bearing = roles[i].Mark || roles[i].Form is { } f && own.JoinedForms.Contains(f) ? 0 : metrics.SideBearing;
            var (minX, _, _, _) = outlines[i].Bounds();
            var outline = GlyphEncoder.Whole(Shifted(outlines[i], bearing - minX), tolerance);
            placed[i] = (outline, Math.Clamp((int)MathF.Round(outline.Bounds().MaxX + bearing), 0, MadeFontFaceBuilder.MostAdvance));
        }

        // The space: as set, else half the median advance of the plain characters written (of every glyph when none is).
        if (metrics.Space < 0)
        {
            var plain = new List<double>();
            var every = new List<double>();
            for (int i = 0; i < found.Count; i++)
            {
                if (!outlines[i].IsEmpty && !roles[i].Mark)
                {
                    (roles[i].Codes.Length == 1 && roles[i].Form is null ? plain : every).Add(placed[i].Advance);
                }
            }

            metrics = metrics with { Space = (int)Math.Round(InkMeasure.Median(plain.Count > 0 ? plain : every.Count > 0 ? every : new List<double> { own.UnitsPerEm / 2.0 }) / 2) };
        }

        // The glyphs by role, in the order their first was found: the first fill, then the others.
        var groups = new Dictionary<(string Text, string? Form), List<int>>();
        var order = new List<(string Text, string? Form)>();
        for (int i = 0; i < found.Count; i++)
        {
            if (outlines[i].IsEmpty)
            {
                leftOut.Add(new GlyphLeftOut(found[i].Text, found[i].Form, "its ink traced to no outline"));
                continue;
            }

            var key = (roles[i].Text, roles[i].Form);
            if (!groups.TryGetValue(key, out var fills))
            {
                groups[key] = fills = [];
                order.Add(key);
            }

            fills.Add(i);
        }

        // Which roles a font map can shape: plain characters and marks, a plain letter's positional forms, ligatures whose
        // letters have the forms they take.
        var usable = new Dictionary<(string Text, string? Form), string?>();       // the role and why it is not usable (null: it is)
        foreach (var key in order)
        {
            var role = roles[groups[key][0]];
            usable[key] = role.Codes.Length == 1
                ? role.Form is null ? null
                : role.Mark ? "a mark takes no joining form"
                : !FontMap.Forms.Contains(role.Form) ? $"'{role.Form}' is not a positional form a font map shapes ({string.Join(", ", FontMap.Forms)})"
                : groups.ContainsKey((role.Text, null)) ? null : "the character has no plain glyph its forms are substituted from"
                : null;
        }

        var components = new Dictionary<(string Text, string? Form), FontMapComponent[]>();
        foreach (var key in order.Where(k => roles[groups[k][0]].Codes.Length > 1))
        {
            var role = roles[groups[key][0]];
            var (parts, why) = Components(role, groups, roles);
            usable[key] = why;
            if (parts is not null)
            {
                components[key] = parts;
            }
        }

        // The face: each usable role's fills added (the plain glyph of a character by its code, every other glyph by id).
        var family = request.Family;
        var builder = new MadeFontFaceBuilder(family, new FontMetrics(upem, metrics.Ascender, metrics.Descender, metrics.LineGap)).Heights(metrics.XHeight, metrics.CapHeight);
        if (request.Style is { } style)
        {
            builder.Name(FontNameKind.Style, style);
        }

        foreach (var (kind, value) in request.Names ?? new Dictionary<FontNameKind, string>())
        {
            builder.Name(kind, value);
        }

        var ids = new Dictionary<(string Text, string? Form), List<int>>();
        var made = new List<(int Index, MadeGlyph Glyph)>(found.Count);
        foreach (var key in order)
        {
            var fills = groups[key];
            if (usable[key] is { } why)
            {
                fills.ForEach(i => leftOut.Add(new GlyphLeftOut(found[i].Text, found[i].Form, why)));
                continue;
            }

            var role = roles[fills[0]];
            var glyphs = new List<int>(fills.Count);
            for (int k = 0; k < fills.Count; k++)
            {
                int i = fills[k];
                if (k > 0 && !own.Alternates)
                {
                    leftOut.Add(new GlyphLeftOut(found[i].Text, found[i].Form, "a further fill: alternates are off, the first fill is kept"));
                    continue;
                }

                int glyph = k == 0 && role.Codes.Length == 1 && role.Form is null
                    ? builder.Add(role.Codes[0], placed[i].Outline, placed[i].Advance)
                    : builder.Add(placed[i].Outline, placed[i].Advance);
                glyphs.Add(glyph);
                made.Add((i, new MadeGlyph(glyph, found[i].Text, found[i].Form, k)));
            }

            ids[key] = glyphs;
        }

        // A space when none was written: no ink, its advance measured (or set); a no-break space shares it.
        if (!ids.ContainsKey((" ", null)))
        {
            int space = builder.Add(' ', GlyphOutline.Empty, metrics.Space);
            if (!ids.ContainsKey(("\u00A0", null)))
            {
                builder.Map(0xA0, space);
            }
        }

        var face = builder.Build();
        var map = FontMap.Parse(MapJson(request, own, metrics, order, roles, groups, ids, components, inks), $"{family} (made)");
        made.Sort((a, b) => a.Index.CompareTo(b.Index));
        return new MadeFont(Apply(face, map), map, metrics, [.. made.Select(m => m.Glyph)], leftOut);
    }

    // The face's metrics from the ink of the glyphs that are not marks (all of them when every glyph is a mark).
    private static MadeFontMetrics MeasureInk(Settings own, Role[] roles, InkStats?[] inks)
    {
        var measured = new List<InkStats>();
        for (int pass = 0; pass < 2 && measured.Count == 0; pass++)
        {
            for (int i = 0; i < inks.Length; i++)
            {
                if (inks[i] is { } ink && (pass == 1 || !roles[i].Mark))
                {
                    measured.Add(ink);
                }
            }
        }

        if (measured.Count == 0)
        {
            throw new ArgumentException("A font is made from glyphs with ink; none of the glyphs given traced to an outline with ink.");
        }

        int upem = own.UnitsPerEm;
        var heights = Heights(measured, own.InkShare);
        int pen = (int)Math.Round(InkMeasure.Median([.. measured.Select(i => i.Thickness)]));
        int bearing = own.SideBearing is { } share ? (int)Math.Round(share * upem) : (int)Math.Round(pen / 2.0);
        int space = own.Space is { } spaceShare ? (int)Math.Round(spaceShare * upem) : -1;
        string described = string.Create(CultureInfo.InvariantCulture,
            $"from the ink of {measured.Count} glyphs at 256 px per em: the ascender and descender hold {own.InkShare:0.##} of their tops and bottoms; the x-height and cap height are the median tops of the {heights.Short} short and {heights.Tall} tall glyphs of at least half the median ink height (Otsu's split at {heights.Level:0.###} of the tallest); the pen is their strokes' median thickness; the side bearing {(own.SideBearing is null ? "half the pen" : "as set")}; the space {(own.Space is null ? "half the plain characters' median advance" : "as set")}");
        return new MadeFontMetrics(upem, Math.Max(1, (int)Math.Ceiling(heights.Ascender)), Math.Min(0, (int)Math.Floor(heights.Descender)), (int)Math.Round(own.LineGap * upem),
            Math.Clamp((int)Math.Round(heights.XHeight), 0, MadeFontFaceBuilder.MostCoordinate), Math.Clamp((int)Math.Round(heights.CapHeight), 0, MadeFontFaceBuilder.MostCoordinate),
            pen, bearing, space, described);
    }

    /// <summary>
    /// The heights of glyphs' ink (font units, y up): the ascender and descender holding <paramref name="share"/> of the
    /// tops and bottoms (the rest outliers); the x-height and cap height, the median tops of the short and the tall glyphs
    /// among those at least half the median ink height (a dot, a comma or a dash is neither), split by Otsu's level of their
    /// tops as a share of the tallest; how many were short and tall, and that level.
    /// </summary>
    internal static (double Ascender, double Descender, double XHeight, double CapHeight, int Short, int Tall, float Level) Heights(IReadOnlyList<InkStats> inks, double share)
    {
        var tops = inks.Select(i => i.MaxY).ToList();
        var bottoms = inks.Select(i => i.MinY).ToList();
        double half = InkMeasure.Median([.. inks.Select(i => i.Height)]) / 2;
        var above = inks.Where(i => i.MaxY > 0 && i.Height >= half).Select(i => i.MaxY).ToList();
        if (above.Count == 0)
        {
            return (Share(tops, share), Share(bottoms, 1 - share), 0, 0, 0, 0, 0);
        }

        double tallest = above.Max();
        float level = ImageThresholds.Otsu([.. above.Select(t => (float)(t / tallest))]);
        var shortTops = above.Where(t => t / tallest < level).ToList();
        var tallTops = above.Where(t => t / tallest >= level).ToList();
        return (Share(tops, share), Share(bottoms, 1 - share), InkMeasure.Median(shortTops.Count > 0 ? shortTops : [.. above]), InkMeasure.Median(tallTops.Count > 0 ? tallTops : [.. above]),
            shortTops.Count, tallTops.Count, level);
    }

    // The value at a share of the sorted values (the nearest rank).
    private static double Share(List<double> values, double share)
    {
        var sorted = values.ToArray();
        Array.Sort(sorted);
        return sorted[Math.Clamp((int)Math.Ceiling(share * sorted.Length) - 1, 0, sorted.Length - 1)];
    }

    // A ligature's components, each its letter in the form it takes inside the whole (by the ligature's form: a name the
    // rule reads, or each component's form between '+'; none: the letters' plain glyphs); null and why when one of its
    // letters has no plain glyph or not that form.
    private static (FontMapComponent[]? Parts, string? Why) Components(Role role, Dictionary<(string Text, string? Form), List<int>> groups, Role[] roles)
    {
        int n = role.Codes.Length;
        var forms = new string?[n];
        if (role.Form is { } form)
        {
            if (form.Contains('+', StringComparison.Ordinal))
            {
                var each = form.Split('+');
                if (each.Length != n)
                {
                    return (null, $"its form '{form}' names {each.Length} forms for {n} letters");
                }

                each.CopyTo(forms, 0);
            }
            else if (LigatureForms.TryGetValue(form, out var rule))
            {
                for (int c = 0; c < n; c++)
                {
                    forms[c] = c == 0 ? rule.First : c == n - 1 ? rule.Last : rule.Middle;
                }
            }
            else
            {
                return (null, $"'{form}' is not a positional form a font map shapes ({string.Join(", ", FontMap.Forms)})");
            }
        }

        var parts = new FontMapComponent[n];
        for (int c = 0; c < n; c++)
        {
            string letter = char.ConvertFromUtf32(role.Codes[c]);
            string? part = forms[c] == "isol" ? null : forms[c];
            if (part is not null && !FontMap.Forms.Contains(part))
            {
                return (null, $"'{forms[c]}' is not a positional form a font map shapes ({string.Join(", ", FontMap.Forms)})");
            }

            if (!groups.TryGetValue((letter, null), out var plain) || roles[plain[0]].Mark)
            {
                return (null, $"its letter U+{role.Codes[c]:X4} has no plain glyph");
            }

            if (part is not null && !groups.ContainsKey((letter, part)))
            {
                return (null, $"its letter U+{role.Codes[c]:X4} has no {part} form");
            }

            parts[c] = new FontMapComponent(role.Codes[c], forms[c]);
        }

        return (parts, null);
    }

    // The font map of the face, as JSON: the letters by their plain glyph and forms, the marks with their place, the
    // ligatures, the alternates, the anchors and the characters measured for the drawn glyphs.
    private static ReadOnlyMemory<byte> MapJson(FontMakerRequest request, Settings own, MadeFontMetrics metrics, List<(string Text, string? Form)> order, Role[] roles,
        Dictionary<(string Text, string? Form), List<int>> groups, Dictionary<(string Text, string? Form), List<int>> ids,
        Dictionary<(string Text, string? Form), FontMapComponent[]> components, InkStats?[] inks)
    {
        static string Id(int glyph) => "#" + glyph.ToString(CultureInfo.InvariantCulture);
        static string Code(int code) => string.Create(CultureInfo.InvariantCulture, $"U+{code:X4}");

        var buffer = new ArrayBufferWriter<byte>(4096);
        using (var w = new Utf8JsonWriter(buffer))
        {
            w.WriteStartObject();
            w.WriteString("name", $"{request.Family} {request.Style ?? "Regular"} (made)");
            w.WriteString("summary", $"the forms, ligatures, marks and alternates of a font made from glyphs: {request.Family}");
            w.WriteStartArray("scripts");
            foreach (string script in own.Scripts)
            {
                w.WriteStringValue(script);
            }

            w.WriteEndArray();

            bool IsMark((string Text, string? Form) key) => roles[groups[key][0]].Mark;
            w.WriteStartObject("letters");
            int letters = 0;
            foreach (var key in order.Where(k => k.Form is null && ids.ContainsKey(k) && roles[groups[k][0]].Codes.Length == 1 && !IsMark(k)))
            {
                w.WriteStartObject(Code(roles[groups[key][0]].Codes[0]));
                w.WriteString("isol", Id(ids[key][0]));
                foreach (string form in FontMap.Forms.Skip(1))
                {
                    if (ids.TryGetValue((key.Text, form), out var formed))
                    {
                        w.WriteString(form, Id(formed[0]));
                    }
                }

                w.WriteEndObject();
                letters++;
            }

            w.WriteEndObject();
            if (letters == 0)
            {
                throw new ArgumentException("A font is made from characters: the glyphs given hold marks and nothing else (a font map shapes letters).");
            }

            var marks = order.Where(k => ids.ContainsKey(k) && IsMark(k)).ToList();
            if (marks.Count > 0)
            {
                w.WriteStartObject("marks");
                foreach (var key in marks)
                {
                    // Above the base when its ink's middle sat over half the x-height, else below it.
                    var ink = inks[groups[key][0]];
                    bool above = ink is { } i && (i.MinY + i.MaxY) / 2 >= metrics.XHeight / 2.0;
                    w.WriteStartObject(Code(roles[groups[key][0]].Codes[0]));
                    w.WriteString("glyph", Id(ids[key][0]));
                    w.WriteString("place", above ? "above" : "below");
                    w.WriteEndObject();
                }

                w.WriteEndObject();
            }

            var ligatures = order.Where(k => ids.ContainsKey(k) && components.ContainsKey(k)).ToList();
            if (ligatures.Count > 0)
            {
                w.WriteStartArray("ligatures");
                foreach (var key in ligatures)
                {
                    w.WriteStartObject();
                    w.WriteString("feature", own.LigatureFeature);
                    w.WriteStartArray("components");
                    foreach (var part in components[key])
                    {
                        w.WriteStringValue(part.ToString());
                    }

                    w.WriteEndArray();
                    w.WriteString("glyph", Id(ids[key][0]));
                    w.WriteEndObject();
                }

                w.WriteEndArray();
            }

            var alternates = order.Where(k => ids.TryGetValue(k, out var g) && g.Count > 1).ToList();
            if (alternates.Count > 0)
            {
                w.WriteStartObject("alternates");
                foreach (var key in alternates)
                {
                    w.WriteStartArray(Id(ids[key][0]));
                    foreach (int glyph in ids[key].Skip(1))
                    {
                        w.WriteStringValue(Id(glyph));
                    }

                    w.WriteEndArray();
                }

                w.WriteEndObject();
            }

            w.WriteStartObject("anchors");
            w.WriteNumber("gap", (int)Math.Round(own.MarkGap * own.UnitsPerEm));
            w.WriteNumber("window", Math.Max(1, (int)Math.Round(own.MarkWindow * own.UnitsPerEm)));
            w.WriteEndObject();
            if (own.Measure.Length > 0)
            {
                w.WriteStartArray("measure");
                foreach (int code in own.Measure)
                {
                    w.WriteStringValue(Code(code));
                }

                w.WriteEndArray();
            }

            w.WriteEndObject();
        }

        return buffer.WrittenMemory.ToArray();
    }

    // The outline moved right by dx font units.
    private static GlyphOutline Shifted(GlyphOutline outline, float dx)
    {
        var points = outline.Points.ToArray();
        for (int i = 0; i < points.Length; i += 2)
        {
            points[i] += dx;
        }

        return new GlyphOutline(outline.Verbs.ToArray(), points);
    }

    // A glyph's role: its text (composed), its code points, its form (null: plain) and whether it is a combining mark.
    private readonly record struct Role(string Text, int[] Codes, string? Form, bool Mark);

    // The settings read once from a maker's options.
    private sealed class Settings : FontMakerSettings
    {
        public Settings(IReadOnlyDictionary<string, string> options) : base(options)
        {
            foreach (var (raw, value) in options)
            {
                string key = raw.ToLowerInvariant();
                string what = $"The font maker traced's {key}";
                switch (key)
                {
                    case UnitsKey:
                        UnitsPerEm = PageOptions.Integer(what, value, 16, 16_384);
                        break;
                    case SideBearingKey:
                        SideBearing = Auto(value) ? null : PageOptions.Number($"{what} (auto, or a share of the em)", value, 0f, 0.5f);
                        break;
                    case JoinedFormsKey:
                        JoinedForms = value.Trim().Equals("none", StringComparison.OrdinalIgnoreCase) ? [] : [.. PageOptions.List($"{what} (forms between ':', or none)", value)];
                        break;
                    case InkShareKey:
                        InkShare = PageOptions.Number(what, value, 0.5f, 1f);
                        break;
                    case LineGapKey:
                        LineGap = PageOptions.Number($"{what} (a share of the em)", value, 0f, 2f);
                        break;
                    case SpaceKey:
                        Space = Auto(value) ? null : PageOptions.Number($"{what} (auto, or a share of the em)", value, 0f, 2f);
                        break;
                    case ScriptsKey:
                        Scripts = [.. PageOptions.List($"{what} (OpenType script tags between ':')", value).Select(t => Tag(what, t))];
                        break;
                    case LigatureFeatureKey:
                        LigatureFeature = Tag(what, value.Trim());
                        break;
                    case AlternatesKey:
                        Alternates = PageOptions.Switch(what, value);
                        break;
                    case MeasureKey:
                        Measure = [.. PageOptions.List($"{what} (characters between ':', each U+XXXX or one character)", value).Select(c => Character(what, c))];
                        break;
                    case MarkGapKey:
                        MarkGap = PageOptions.Number($"{what} (a share of the em)", value, 0f, 1f);
                        break;
                    case MarkWindowKey:
                        MarkWindow = PageOptions.Number($"{what} (a share of the em)", value, 0.01f, 2f);
                        break;
                    case ThreadsKey:
                        Threads = Auto(value) ? 0 : PageOptions.Integer($"{what} (auto, or a number)", value, 1, 1024);
                        break;
                    default:
                        throw new ArgumentException($"The font maker traced does not take '{raw}'; it takes: {Own}.", nameof(options));
                }
            }
        }

        public int UnitsPerEm { get; } = 1000;

        public float? SideBearing { get; }

        public IReadOnlyList<string> JoinedForms { get; } = ["init", "medi", "fina"];

        public float InkShare { get; } = 0.95f;

        public float LineGap { get; } = 0.2f;

        public float? Space { get; }

        public IReadOnlyList<string> Scripts { get; } = ["DFLT"];

        public string LigatureFeature { get; } = "liga";

        public bool Alternates { get; } = true;

        public int[] Measure { get; } = [];

        public float MarkGap { get; } = 0.06f;

        public float MarkWindow { get; } = 0.13f;

        public int Threads { get; }

        private static bool Auto(string value) => value.Trim().Equals("auto", StringComparison.OrdinalIgnoreCase);

        // An OpenType tag: four printable ASCII characters.
        private static string Tag(string what, string tag) =>
            tag.Length == 4 && tag.All(c => c is >= ' ' and <= '~') ? tag : throw new ArgumentException($"{what} holds '{tag}': an OpenType tag is four printable characters (DFLT, liga).");

        // A character: U+XXXX or the character itself.
        private static int Character(string what, string text)
        {
            if (FontMap.TryCodePoint(text, out int code))
            {
                return code;
            }

            var runes = text.EnumerateRunes();
            return runes.MoveNext() && runes.Current.Utf16SequenceLength == text.Length
                ? runes.Current.Value
                : throw new ArgumentException($"{what} holds '{text}': a character is U+XXXX or one character.");
        }
    }
}
