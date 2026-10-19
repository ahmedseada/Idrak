// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using Idrak.Abstraction.Text;

namespace Idrak.Imaging.Shaping;

/// <summary>A font's layout tables, read once per face (kept with it).</summary>
internal sealed class FontLayout
{
    private static readonly ConditionalWeakTable<IFontFace, FontLayout> Cache = [];

    private FontLayout(IFontFace font)
    {
        Definitions = new GlyphDefinitions(font.Table("GDEF"));
        Substitutions = new LayoutTable(font.Table("GSUB"), positioning: false);
        Positions = new LayoutTable(font.Table("GPOS"), positioning: true);
        SpaceGlyph = font.Glyph(' ');
    }

    public static FontLayout For(IFontFace font) => Cache.GetValue(font, f => new FontLayout(f));

    public GlyphDefinitions Definitions { get; }

    public LayoutTable Substitutions { get; }

    public LayoutTable Positions { get; }

    public int SpaceGlyph { get; }

    /// <summary>Plans by script, language, shaper and features, made once each.</summary>
    public ConcurrentDictionary<(string Script, string? Language, string Shaper, string Features), ShapingPlan> Plans { get; } = new();
}

/// <summary>A feature a shaper asks for: in which GSUB stage, whether every glyph gets it, and whether ZWJ / ZWNJ are matched by hand.</summary>
internal readonly record struct FeatureRequest(string Tag, int Stage, bool Global = true, bool ManualZwj = false, bool ManualZwnj = false);

/// <summary>The lookups to run for one script, language and set of features: GSUB by stage, then GPOS, each with its mask.</summary>
internal sealed class ShapingPlan
{
    public required List<List<(Lookup Lookup, uint Mask, bool AutoZwj, bool AutoZwnj)>> SubstitutionStages { get; init; }

    public required List<(Lookup Lookup, uint Mask)> PositionLookups { get; init; }

    public required uint GlobalMask { get; init; }

    public required Dictionary<string, uint> Masks { get; init; }

    public required bool HasPositioning { get; init; }

    public required bool HasMarkPositioning { get; init; }

    public required HashSet<string> SubstitutionFeatures { get; init; }
}

/// <summary>
/// "general", and the path every shaper of the library follows (HarfBuzz's OpenType shaping), for any script:
/// normalization by what the font has, mirroring in right-to-left text, the character map, GDEF classes (or classes from
/// the characters), the shaper's masks, GSUB by stage, default advances, GPOS, mark widths zeroed late, attachment
/// offsets, default ignorables hidden, and a right-to-left run reversed. As "general" it shapes every script without a
/// shaper of its own (Latin, Greek, Cyrillic, Armenian, Georgian, Hebrew, the CJK scripts, ...) with HarfBuzz's default
/// features; a script's own shaper (Arabic) overrides its hooks.
/// </summary>
internal class OpenTypeShaper : ITextShaper
{
    public virtual string Name => "general";

    // The GSUB features of the shaper in stage order; GPOS features need no stages.
    protected virtual IReadOnlyList<FeatureRequest> SubstitutionFeatures(bool rightToLeft) =>
    [
        new("rvrn", 0),
        .. rightToLeft ? (FeatureRequest[])[new("rtla", 1), new("rtlm", 1)] : [new("ltra", 1), new("ltrm", 1)],
        new("ccmp", 1), new("locl", 1), new("rlig", 1, ManualZwj: true), new("calt", 1, ManualZwj: true), new("clig", 1), new("liga", 1), new("rclt", 1, ManualZwj: true),
    ];

    protected virtual IReadOnlyList<string> PositionFeatures(bool rightToLeft) => ["abvm", "blwm", "mark", "mkmk", "curs", "dist", "kern"];

    /// <summary>Per-glyph masks the shaper sets (Arabic's positional forms); the global mask is set already.</summary>
    protected virtual void SetupMasks(ShapingPlan plan, List<ShapingGlyph> glyphs, string text, TextRun run) { }

    /// <summary>A shaper's own reordering of a run of marks (Arabic's modifier combining marks).</summary>
    protected virtual void ReorderMarks(List<ShapingGlyph> glyphs, int start, int end) { }

    /// <summary>A shaper's own substitution when the font has no GSUB for the script (Arabic's presentation forms).</summary>
    protected virtual void FallbackSubstitution(IFontFace font, ShapingPlan plan, List<ShapingGlyph> glyphs, string text, TextRun run) { }

    public ShapedGlyph[] Shape(IFontFace font, string text, TextRun run, IReadOnlyCollection<string>? features = null)
    {
        ArgumentNullException.ThrowIfNull(font);
        ArgumentNullException.ThrowIfNull(text);
        if (run.Start < 0 || run.Length < 0 || run.Start + run.Length > text.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(run), $"The run ({run.Start}, {run.Length}) lies outside the text's {text.Length} characters.");
        }

        if (run.Length == 0)
        {
            return [];
        }

        // A run against its script's own direction (Latin set right to left, Arabic left to right) is shaped as HarfBuzz
        // does: its grapheme clusters reversed, then in the script's direction.
        if (NativeDirection(run.Script, text, run) is bool native && native != run.RightToLeft)
        {
            string reversed = ReverseGraphemes(text.Substring(run.Start, run.Length), out int[] origins);
            var inner = Shape(font, reversed, run with { Start = 0, Length = reversed.Length, RightToLeft = native }, features);
            for (int i = 0; i < inner.Length; i++)
            {
                inner[i] = inner[i] with { Cluster = run.Start + origins[inner[i].Cluster] };
            }

            return inner;                                                            // already in drawing order
        }

        var layout = FontLayout.For(font);
        string featureKey = features is null ? "" : string.Join(",", features);
        var plan = layout.Plans.GetOrAdd((run.Script, run.Language, Name + (run.RightToLeft ? ":rtl" : ":ltr"), featureKey),
            key => MakePlan(layout, run, features));

        var glyphs = Characters(font, text, run, layout);
        foreach (ref var g in CollectionsMarshal.AsSpan(glyphs))
        {
            g.Mask = plan.GlobalMask;
        }

        SetupMasks(plan, glyphs, text, run);

        var context = new LayoutContext(font, layout.Definitions, glyphs, run.RightToLeft);
        if (plan.SubstitutionFeatures.Count == 0)
        {
            FallbackSubstitution(font, plan, glyphs, text, run);
        }

        var substitute = new LayoutEngine(context, layout.Substitutions, positioning: false);
        foreach (var stage in plan.SubstitutionStages)
        {
            foreach (var (lookup, mask, autoZwj, autoZwnj) in stage)
            {
                substitute.Apply(lookup, mask, autoZwj, autoZwnj);
            }
        }

        // Default positions: each glyph's advance.
        foreach (ref var g in CollectionsMarshal.AsSpan(glyphs))
        {
            g.XAdvance = font.Advance(g.Glyph);
            g.YAdvance = 0;
            g.XOffset = 0;
            g.YOffset = 0;
        }

        var position = new LayoutEngine(context, layout.Positions, positioning: true);
        foreach (var (lookup, mask) in plan.PositionLookups)
        {
            position.Apply(lookup, mask);
        }

        var span = CollectionsMarshal.AsSpan(glyphs);
        foreach (ref var g in span)
        {
            if (g.IsMark)
            {
                g.XAdvance = 0;                                                     // marks take no width (by GDEF, after GPOS)
                g.YAdvance = 0;
            }

            if (g.Ignorable)
            {
                g.XAdvance = 0;
            }
        }

        for (int i = 0; i < span.Length; i++)
        {
            Propagate(span, i, run.RightToLeft, 64);
        }

        var result = new ShapedGlyph[span.Length];
        for (int i = 0; i < span.Length; i++)
        {
            var g = span[i];
            int glyph = g.Ignorable ? (layout.SpaceGlyph != 0 ? layout.SpaceGlyph : g.Glyph) : g.Glyph;
            result[i] = new ShapedGlyph(glyph, g.Cluster, g.XAdvance, g.YAdvance, g.Ignorable ? 0 : g.XOffset, g.Ignorable ? 0 : g.YOffset);
        }

        if (run.RightToLeft)
        {
            Array.Reverse(result);
        }

        return result;
    }

    // Whether a script reads right to left by nature (null: no direction of its own, as digits and punctuation).
    private static bool? NativeDirection(string script, string text, TextRun run)
    {
        if (script is "Zyyy" or "Zinh" or "Zzzz" || script.Length != 4)
        {
            return null;
        }

        bool rtl = RightToLeftScripts.Contains(script);
        if (rtl && !run.RightToLeft)
        {
            // Digits in a right-to-left script's run set left to right are left to right by nature.
            bool numeric = true;
            for (int i = run.Start; i < run.Start + run.Length && numeric; i++)
            {
                numeric = char.IsDigit(text[i]) || char.GetUnicodeCategory(text[i]) is System.Globalization.UnicodeCategory.DecimalDigitNumber;
            }

            if (numeric)
            {
                return false;
            }
        }

        return rtl;
    }

    private static readonly HashSet<string> RightToLeftScripts = new(StringComparer.OrdinalIgnoreCase)
    {
        "Arab", "Hebr", "Syrc", "Thaa", "Cprt", "Khar", "Phnx", "Nkoo", "Lydi", "Avst", "Armi", "Phli", "Prti", "Sarb", "Orkh", "Samr", "Mand",
        "Merc", "Mero", "Mani", "Mend", "Nbat", "Narb", "Palm", "Phlp", "Hatr", "Hung", "Adlm", "Rohg", "Sogo", "Sogd", "Elym", "Chrs", "Yezi",
        "Ougr",
    };

    // The text with its grapheme clusters (a character and the marks and joiners after it) in reverse order, and for
    // each new index the old one.
    private static string ReverseGraphemes(string text, out int[] origins)
    {
        var clusters = new List<(int Start, int Length)>();
        for (int i = 0; i < text.Length;)
        {
            int start = i;
            UnicodeText.CodePointAt(text, i, out int w);
            i += w;
            while (i < text.Length)
            {
                int cp = UnicodeText.CodePointAt(text, i, out int w2);
                if (!(UnicodeText.IsMark(cp) || cp is 0x200C or 0x200D))
                {
                    break;
                }

                i += w2;
            }

            clusters.Add((start, i - start));
        }

        var builder = new StringBuilder(text.Length);
        origins = new int[text.Length];
        for (int c = clusters.Count - 1; c >= 0; c--)
        {
            var (start, length) = clusters[c];
            for (int k = 0; k < length; k++)
            {
                origins[builder.Length] = start + k;
                builder.Append(text[start + k]);
            }
        }

        return builder.ToString();
    }

    // An attached glyph's offset adds up its parent's (marks: and the advances between them).
    private static void Propagate(Span<ShapingGlyph> g, int i, bool rightToLeft, int depth)
    {
        int chain = g[i].AttachChain;
        if (chain == 0)
        {
            return;
        }

        var kind = g[i].Attach;
        g[i].AttachChain = 0;
        int j = i + chain;
        if ((uint)j >= (uint)g.Length || depth == 0)
        {
            return;
        }

        Propagate(g, j, rightToLeft, depth - 1);
        if (kind == AttachKind.Cursive)
        {
            g[i].YOffset += g[j].YOffset;
            return;
        }

        g[i].XOffset += g[j].XOffset;
        g[i].YOffset += g[j].YOffset;
        if (!rightToLeft)
        {
            for (int k = j; k < i; k++)
            {
                g[i].XOffset -= g[k].XAdvance;
            }
        }
        else
        {
            for (int k = j + 1; k < i + 1; k++)
            {
                g[i].XOffset += g[k].XAdvance;
            }
        }
    }

    // The run's characters as glyphs: normalized by what the font has, marks reordered, mirrored when right to left,
    // mapped, and classed.
    private List<ShapingGlyph> Characters(IFontFace font, string text, TextRun run, FontLayout layout)
    {
        var chars = new List<(int CodePoint, int Cluster)>(run.Length);
        for (int i = run.Start; i < run.Start + run.Length;)
        {
            int cp = UnicodeText.CodePointAt(text, i, out int width);
            chars.Add((cp, i));
            i += width;
        }

        // A letter and what continues it (marks, ZWJ, emoji modifiers, tag characters) are one cluster, the letter's, as
        // HarfBuzz forms them before shaping (hb_form_clusters) at its default level; at character level each keeps its own.
        for (int k = 1, start = 0; run.Clusters == TextClusters.Graphemes && k < chars.Count; k++)
        {
            int c = chars[k].CodePoint;
            if (UnicodeText.IsMark(c) || c == 0x200D || c is >= 0x1F3FB and <= 0x1F3FF or 0xFF9E or 0xFF9F or >= 0xE0020 and <= 0xE007F)
            {
                chars[k] = (c, chars[start].Cluster);
            }
            else
            {
                start = k;
            }
        }

        // Decompose what the font has no glyph for, when it has the parts.
        var decomposed = new List<(int CodePoint, int Cluster)>(chars.Count);
        foreach (var (cp, cluster) in chars)
        {
            if (font.Glyph(cp) == 0 && !UnicodeText.IsDefaultIgnorable(cp) && Decompose(font, cp) is { } parts)
            {
                foreach (int part in parts)
                {
                    decomposed.Add((part, cluster));
                }
            }
            else
            {
                decomposed.Add((cp, cluster));
            }
        }

        var glyphs = new List<ShapingGlyph>(decomposed.Count);
        foreach (var (cp, cluster) in decomposed)
        {
            glyphs.Add(new ShapingGlyph
            {
                CodePoint = cp, Cluster = cluster, Combining = (byte)ModifiedCombiningClass(cp), Ignorable = UnicodeText.IsDefaultIgnorable(cp), Components = 1,
            });
        }

        // Marks in canonical order (HarfBuzz's modified classes: shadda first in Arabic), then the shaper's own order.
        var span = CollectionsMarshal.AsSpan(glyphs);
        for (int i = 0; i < span.Length;)
        {
            if (span[i].Combining == 0)
            {
                i++;
                continue;
            }

            int end = i;
            while (end < span.Length && span[end].Combining != 0)
            {
                end++;
            }

            if (end - i > 1)
            {
                SortMarks(span, i, end);
                ReorderMarks(glyphs, i, end);
                span = CollectionsMarshal.AsSpan(glyphs);
            }

            i = end;
        }

        // Compose a base and a following mark the font has a glyph for together (not across a blocking mark).
        for (int i = 0; i < glyphs.Count; i++)
        {
            if (glyphs[i].Combining != 0)
            {
                continue;
            }

            int lastClass = -1;
            for (int j = i + 1; j < glyphs.Count && glyphs[j].Combining != 0;)
            {
                int cls = glyphs[j].Combining;
                if (cls > lastClass && Compose(glyphs[i].CodePoint, glyphs[j].CodePoint) is int composite && font.Glyph(composite) != 0)
                {
                    var head = glyphs[i];
                    head.CodePoint = composite;
                    glyphs[i] = head;
                    glyphs.RemoveAt(j);
                    continue;
                }

                lastClass = Math.Max(lastClass, cls);
                j++;
            }
        }

        // Mirror brackets in right-to-left text when the font has the mirror.
        span = CollectionsMarshal.AsSpan(glyphs);
        if (run.RightToLeft)
        {
            foreach (ref var g in span)
            {
                int mirror = UnicodeText.Mirror(g.CodePoint);
                if (mirror != g.CodePoint && font.Glyph(mirror) != 0)
                {
                    g.CodePoint = mirror;
                }
            }
        }

        var definitions = layout.Definitions;
        foreach (ref var g in span)
        {
            g.Glyph = font.Glyph(g.CodePoint);
            if (definitions.HasClasses)
            {
                g.Kind = definitions.Kind(g.Glyph);
                g.MarkClass = definitions.MarkClass(g.Glyph);
            }
            else
            {
                g.Kind = !g.Ignorable && char.GetUnicodeCategory(char.ConvertFromUtf32(g.CodePoint), 0) == System.Globalization.UnicodeCategory.NonSpacingMark
                    ? GlyphKind.Mark : GlyphKind.Base;
            }
        }

        return glyphs;
    }

    // HarfBuzz's stable insertion sort by combining class; a mark that moves shares one cluster with what it passed.
    private static void SortMarks(Span<ShapingGlyph> g, int start, int end)
    {
        for (int i = start + 1; i < end; i++)
        {
            int j = i;
            while (j > start && g[j - 1].Combining > g[i].Combining)
            {
                j--;
            }

            if (j == i)
            {
                continue;
            }

            var moving = g[i];
            g[j..i].CopyTo(g[(j + 1)..(i + 1)]);
            g[j] = moving;
            MergeClusters(g, j, i + 1);
        }
    }

    /// <summary>
    /// Gives the glyphs [start, end) their smallest cluster, and the glyphs on either side that shared a cluster with
    /// the span's ends too (HarfBuzz's merge_clusters: a mark that shared its letter's cluster follows the letter).
    /// </summary>
    internal static void MergeClusters(Span<ShapingGlyph> g, int start, int end)
    {
        if (end - start < 2)
        {
            return;
        }

        int cluster = int.MaxValue;
        for (int k = start; k < end; k++)
        {
            cluster = Math.Min(cluster, g[k].Cluster);
        }

        if (cluster != g[end - 1].Cluster)
        {
            while (end < g.Length && g[end - 1].Cluster == g[end].Cluster)
            {
                end++;
            }
        }

        if (cluster != g[start].Cluster)
        {
            while (start > 0 && g[start - 1].Cluster == g[start].Cluster)
            {
                start--;
            }
        }

        for (int k = start; k < end; k++)
        {
            g[k].Cluster = cluster;
        }
    }

    private static int[]? Decompose(IFontFace font, int cp)
    {
        string s = char.ConvertFromUtf32(cp);
        string d = UnicodeNormalization.Normalize(s, NormalizationForm.FormD);
        if (d == s)
        {
            return null;
        }

        var parts = new List<int>();
        for (int i = 0; i < d.Length;)
        {
            int p = UnicodeText.CodePointAt(d, i, out int w);
            if (font.Glyph(p) == 0)
            {
                return null;
            }

            parts.Add(p);
            i += w;
        }

        return [.. parts];
    }

    private static int? Compose(int a, int b)
    {
        if (UnicodeText.IsDefaultIgnorable(b))
        {
            return null;
        }

        string s = char.ConvertFromUtf32(a) + char.ConvertFromUtf32(b);
        string c = UnicodeNormalization.Normalize(s, NormalizationForm.FormC);
        if (c.Length == 0 || c == s)
        {
            return null;
        }

        int composite = UnicodeText.CodePointAt(c, 0, out int w);
        return w == c.Length ? composite : null;
    }

    // HarfBuzz's modified combining classes: Hebrew and Arabic points in the order fonts expect (shadda first).
    internal static int ModifiedCombiningClass(int cp)
    {
        int c = UnicodeText.CombiningClass(cp);
        return c switch
        {
            10 => 22, 11 => 15, 12 => 16, 13 => 17, 14 => 23, 15 => 18, 16 => 19, 17 => 20, 18 => 21, 19 => 14, 20 => 24, 21 => 12, 22 => 25,
            23 => 13, 24 => 10, 25 => 11, 26 => 26,
            27 => 28, 28 => 29, 29 => 30, 30 => 31, 31 => 32, 32 => 33, 33 => 27, 34 => 34, 35 => 35,
            _ => c,
        };
    }

    // The language system's first feature of that tag, or -1.
    private static int FirstFeature(LayoutTable table, int[] indices, string tag)
    {
        uint wanted = Bytes.Tag(tag);
        foreach (int i in indices)
        {
            if (table.FeatureTag(i) == wanted)
            {
                return i;
            }
        }

        return -1;
    }

    private ShapingPlan MakePlan(FontLayout layout, TextRun run, IReadOnlyCollection<string>? features)
    {
        var scriptTags = OpenTypeTags.Scripts(run.Script);
        string? languageTag = OpenTypeTags.Language(run.Language);

        // Requested features merged by tag (the earliest stage, global if any asks so), then the caller's changes.
        var requests = new Dictionary<string, FeatureRequest>(StringComparer.Ordinal);
        foreach (var f in SubstitutionFeatures(run.RightToLeft))
        {
            requests[f.Tag] = requests.TryGetValue(f.Tag, out var old)
                ? old with { Stage = Math.Min(old.Stage, f.Stage), Global = old.Global || f.Global, ManualZwj = old.ManualZwj && f.ManualZwj }
                : f;
        }

        int lastStage = requests.Count == 0 ? 0 : requests.Values.Max(f => f.Stage);
        var positionTags = new List<string>(PositionFeatures(run.RightToLeft));
        foreach (string change in features ?? [])
        {
            string tag = change.TrimStart('+', '-');
            if (tag.Length != 4)
            {
                throw new ArgumentException($"'{change}' is not an OpenType feature tag of four characters with + or -.", nameof(features));
            }

            if (change.StartsWith('-'))
            {
                requests.Remove(tag);
                positionTags.Remove(tag);
            }
            else
            {
                requests.TryAdd(tag, new FeatureRequest(tag, lastStage));
                if (!positionTags.Contains(tag))
                {
                    positionTags.Add(tag);
                }
            }
        }

        // A mask bit per feature: the global ones together form the mask every glyph starts with.
        var masks = new Dictionary<string, uint>(StringComparer.Ordinal);
        uint global = 1, next = 2;
        foreach (var f in requests.Values.OrderBy(f => f.Tag, StringComparer.Ordinal))
        {
            if (f.Global)
            {
                masks[f.Tag] = 1;
            }
            else
            {
                masks[f.Tag] = next;
                next <<= 1;
            }
        }

        var stages = new List<List<(Lookup, uint, bool, bool)>>();
        var substitutionFeatures = new HashSet<string>(StringComparer.Ordinal);
        var gsub = layout.Substitutions;
        if (gsub.LanguageSystem(scriptTags, languageTag) is var (indices, required))
        {
            for (int stage = 0; stage <= lastStage; stage++)
            {
                var byLookup = new SortedDictionary<int, (Lookup Lookup, uint Mask, bool AutoZwj, bool AutoZwnj)>();
                if (stage == 0 && required >= 0)
                {
                    foreach (int l in gsub.FeatureLookups(required))
                    {
                        if (l < gsub.Lookups.Length)
                        {
                            byLookup[l] = (gsub.Lookups[l], global, true, true);
                        }
                    }
                }

                foreach (var f in requests.Values.Where(f => f.Stage == stage))
                {
                    int feature = FirstFeature(gsub, indices, f.Tag);
                    if (feature < 0)
                    {
                        continue;
                    }

                    substitutionFeatures.Add(f.Tag);
                    uint mask = masks[f.Tag];
                    foreach (int l in gsub.FeatureLookups(feature))
                    {
                        if (l >= gsub.Lookups.Length)
                        {
                            continue;
                        }

                        byLookup[l] = byLookup.TryGetValue(l, out var old)
                            ? (old.Lookup, old.Mask | mask, old.AutoZwj && !f.ManualZwj, old.AutoZwnj && !f.ManualZwnj)
                            : (gsub.Lookups[l], mask, !f.ManualZwj, !f.ManualZwnj);
                    }
                }

                // A positioning feature the font has in GSUB too is applied there, in the last stage (HarfBuzz looks every
                // feature of its plan up in both tables).
                if (stage == lastStage)
                {
                    foreach (string tag in positionTags.Where(t => !requests.ContainsKey(t)))
                    {
                        int feature = FirstFeature(gsub, indices, tag);
                        foreach (int l in feature < 0 ? [] : gsub.FeatureLookups(feature))
                        {
                            if (l < gsub.Lookups.Length && !byLookup.ContainsKey(l))
                            {
                                byLookup[l] = (gsub.Lookups[l], global, true, true);
                            }
                        }
                    }
                }

                stages.Add([.. byLookup.Values.Select(v => (v.Lookup, v.Mask, v.AutoZwj, v.AutoZwnj))]);
            }
        }

        var positionLookups = new List<(Lookup, uint)>();
        bool hasMark = false;
        var gpos = layout.Positions;
        if (gpos.LanguageSystem(scriptTags, languageTag) is var (posIndices, posRequired))
        {
            var byLookup = new SortedDictionary<int, (Lookup Lookup, uint Mask)>();
            void Add(int feature, uint mask)
            {
                foreach (int l in gpos.FeatureLookups(feature))
                {
                    if (l < gpos.Lookups.Length)
                    {
                        byLookup[l] = byLookup.TryGetValue(l, out var old) ? (old.Lookup, old.Mask | mask) : (gpos.Lookups[l], mask);
                    }
                }
            }

            if (posRequired >= 0)
            {
                Add(posRequired, global);
            }

            foreach (string tag in positionTags)
            {
                int feature = FirstFeature(gpos, posIndices, tag);
                if (feature >= 0)
                {
                    hasMark |= tag is "mark" or "mkmk";
                    Add(feature, global);
                }
            }

            // A substitution feature the font has in GPOS too (Reem Kufi's rclt kerning) is applied there, with its mask
            // (HarfBuzz looks every feature of its plan up in both tables).
            foreach (var f in requests.Values.Where(f => !positionTags.Contains(f.Tag)))
            {
                int feature = FirstFeature(gpos, posIndices, f.Tag);
                if (feature >= 0)
                {
                    Add(feature, masks[f.Tag]);
                }
            }

            positionLookups.AddRange(byLookup.Values.Select(v => (v.Lookup, v.Mask)));
        }

        return new ShapingPlan
        {
            SubstitutionStages = stages.Select(s => s.Select(x => (x.Item1, x.Item2, x.Item3, x.Item4)).ToList()).ToList(),
            PositionLookups = positionLookups, GlobalMask = global, Masks = masks, HasPositioning = positionLookups.Count > 0,
            HasMarkPositioning = hasMark, SubstitutionFeatures = substitutionFeatures,
        };
    }
}

/// <summary>OpenType script and language tags for ISO 15924 scripts and BCP 47 languages.</summary>
internal static class OpenTypeTags
{
    private static readonly Dictionary<string, string[]> Special = new(StringComparer.OrdinalIgnoreCase)
    {
        ["Beng"] = ["bng2", "beng"], ["Deva"] = ["dev2", "deva"], ["Gujr"] = ["gjr2", "gujr"], ["Guru"] = ["gur2", "guru"], ["Knda"] = ["knd2", "knda"],
        ["Mlym"] = ["mlm2", "mlym"], ["Orya"] = ["ory2", "orya"], ["Taml"] = ["tml2", "taml"], ["Telu"] = ["tel2", "telu"], ["Mymr"] = ["mym2", "mymr"],
        ["Hira"] = ["kana"], ["Kana"] = ["kana"], ["Hrkt"] = ["kana"], ["Laoo"] = ["lao "], ["Yiii"] = ["yi  "], ["Nkoo"] = ["nko "], ["Vaii"] = ["vai "],
        ["Zyyy"] = ["DFLT"], ["Zinh"] = ["DFLT"], ["Zzzz"] = ["DFLT"], ["Hang"] = ["hang"], ["Hani"] = ["hani"], ["Copt"] = ["copt"],
    };

    private static readonly Dictionary<string, string> Languages = new(StringComparer.OrdinalIgnoreCase)
    {
        ["ar"] = "ARA ", ["fa"] = "FAR ", ["ur"] = "URD ", ["ps"] = "PAS ", ["sd"] = "SND ", ["ug"] = "UYG ", ["ku"] = "KUR ", ["he"] = "IWR ",
        ["yi"] = "JII ", ["en"] = "ENG ", ["fr"] = "FRA ", ["de"] = "DEU ", ["es"] = "ESP ", ["it"] = "ITA ", ["pt"] = "PTG ", ["tr"] = "TRK ",
        ["ru"] = "RUS ", ["uk"] = "UKR ", ["bg"] = "BGR ", ["sr"] = "SRB ", ["el"] = "ELL ", ["hi"] = "HIN ", ["mr"] = "MAR ", ["ne"] = "NEP ",
        ["bn"] = "BEN ", ["ta"] = "TAM ", ["te"] = "TEL ", ["th"] = "THA ", ["zh"] = "ZHS ", ["ja"] = "JAN ", ["ko"] = "KOR ", ["vi"] = "VIT ",
        ["pl"] = "PLK ", ["nl"] = "NLD ", ["ro"] = "ROM ", ["ms"] = "MLY ", ["id"] = "IND ",
    };

    /// <summary>The OpenType script tags to try for an ISO 15924 code, the newer first.</summary>
    public static IReadOnlyList<string> Scripts(string iso) =>
        Special.TryGetValue(iso, out var tags) ? tags : iso.Length == 4 ? [iso.ToLowerInvariant()] : ["DFLT"];

    /// <summary>The OpenType language tag of a BCP 47 language (its primary subtag), or null for the default.</summary>
    public static string? Language(string? bcp47) =>
        bcp47 is null ? null : Languages.GetValueOrDefault(bcp47.Split('-', '_')[0]);
}
