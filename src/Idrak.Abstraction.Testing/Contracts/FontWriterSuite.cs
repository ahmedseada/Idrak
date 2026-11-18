// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Data;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// The checks of a font writer (<see cref="IFontWriter"/>, registered in <see cref="FontWriters"/>) on faces of its own:
/// each written whole and as subsets, then read back (through <see cref="FontFormats"/> unless a reader is given):
/// <list type="bullet">
/// <item>the same face and options write the same bytes;</item>
/// <item>whole: the metrics, the glyph count, every advance and outline (the same points when <see cref="Exact"/>, else
/// bounds within the curve tolerance), the characters (Latin, Greek, Cyrillic, Hebrew and Arabic code points probed) and
/// the shaping of sample lines through <see cref="TextShapers"/> (where a shaper is registered) are the face's;</item>
/// <item>a subset keeps its glyphs (their ids, outlines and advances, the parts of composites drawn the same) and their
/// characters.</item>
/// </list>
/// The random cases write random subsets.
/// </summary>
public sealed class FontWriterSuite : ContractSuite<IFontWriter>
{
    private static readonly (int First, int Last)[] Probed = [(0x20, 0x7E), (0xA0, 0x24F), (0x370, 0x3FF), (0x400, 0x4FF), (0x590, 0x6FF), (0xFB50, 0xFB60)];

    private static readonly (string Text, string Script, bool RightToLeft)[] Lines =
        [("Thank you for your letter of last week.", "Latn", false), ("The report is attached: 12 pages, 3 figures.", "Latn", false), ("شكرا لرسالتك", "Arab", true)];

    private readonly IReadOnlyList<IFontFace> _faces;
    private readonly Func<ReadOnlyMemory<byte>, IFontFace> _read;
    private readonly FontWriteOptions _options;

    /// <summary>The checks on <paramref name="faces"/>.</summary>
    /// <param name="faces">The faces written (at least one).</param>
    /// <param name="exact">Whether outlines read back with the same points (a writer of the same outline kind as the faces), else within the curve tolerance.</param>
    /// <param name="read">Reads a written file back as a face; null for the first face <see cref="FontFormats"/> reads.</param>
    /// <param name="options">The options faces are written with (a subset's glyphs are the case's); null for the defaults.</param>
    public FontWriterSuite(IReadOnlyList<IFontFace> faces, bool exact = true, Func<ReadOnlyMemory<byte>, IFontFace>? read = null, FontWriteOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(faces);
        _faces = faces.Count > 0 ? faces : throw new ArgumentException("A font writer is checked on faces of its own: give at least one.", nameof(faces));
        Exact = exact;
        _read = read ?? (bytes => FontFormats.Read(bytes, "written")[0]);
        _options = options ?? new FontWriteOptions();
    }

    /// <summary>Whether outlines must read back with the same points.</summary>
    public bool Exact { get; }

    /// <inheritdoc />
    public override string Name => "font-writer";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        for (int f = 0; f < _faces.Count; f++)
        {
            yield return new ContractCase($"{_faces[f].Family} {_faces[f].Style}, whole", new JsonObject { ["face"] = f, ["kind"] = "whole" });
            int[] letters = [.. "Report 2026".EnumerateRunes().Select(r => _faces[f].Glyph(r.Value)).Where(g => g > 0).Distinct()];
            yield return new ContractCase($"{_faces[f].Family} {_faces[f].Style}, the glyphs of a word, with layout", Subset(f, letters, keepLayout: true));
            yield return new ContractCase($"{_faces[f].Family} {_faces[f].Style}, the glyphs of a word, without layout", Subset(f, letters, keepLayout: false));
        }
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        int f = random.Next(_faces.Count);
        int count = Math.Max(1, _faces[f].GlyphCount);
        int[] glyphs = [.. Enumerable.Range(0, 1 + random.Next(Math.Min(count, large ? 400 : 20))).Select(_ => random.Next(count)).Distinct()];
        return new ContractCase($"{_faces[f].Family}, {glyphs.Length} random glyphs", Subset(f, glyphs, keepLayout: random.Next(2) == 0));
    }

    /// <inheritdoc />
    public override void Run(IFontWriter implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var face = _faces[(int)@case.Data["face"]!];
        int[]? kept = @case.Data["glyphs"] is JsonArray list ? [.. list.Select(g => (int)g!)] : null;
        var options = kept is null ? _options with { Glyphs = null } : _options with { Glyphs = kept, KeepLayout = (bool)@case.Data["layout"]! };
        byte[] first = Write(implementation, face, options), second = Write(implementation, face, options);
        checks.Check("the same face writes the same bytes", first.AsSpan().SequenceEqual(second), $"{first.Length} and {second.Length} bytes differ");
        var read = _read(first);
        checks.Check("the metrics", read.Metrics == face.Metrics, $"{read.Metrics}, the face's {face.Metrics}");
        if (kept is null)
        {
            checks.Check("the glyph count", read.GlyphCount == face.GlyphCount, $"{read.GlyphCount} glyphs, the face has {face.GlyphCount}");
            CheckGlyphs(face, read, Enumerable.Range(0, Math.Min(face.GlyphCount, read.GlyphCount)), checks, "every advance and outline");
            CheckCharacters(face, read, null, checks, "the characters");
            CheckShaping(face, read, checks);
            return;
        }

        checks.Check("a subset keeps its glyph ids", read.GlyphCount > kept.DefaultIfEmpty(0).Max(), $"{read.GlyphCount} glyphs, glyph {kept.DefaultIfEmpty(0).Max()} kept");
        CheckGlyphs(face, read, kept.Where(g => g < read.GlyphCount), checks, "a subset's glyphs, their advances and outlines");
        CheckCharacters(face, read, [.. kept], checks, "a subset's characters");
    }

    private static byte[] Write(IFontWriter implementation, IFontFace face, FontWriteOptions options)
    {
        using var stream = new MemoryStream();
        implementation.Write(face, stream, options);
        return stream.ToArray();
    }

    private void CheckGlyphs(IFontFace face, IFontFace read, IEnumerable<int> glyphs, CaseChecks checks, string property)
    {
        foreach (int g in glyphs)
        {
            if (face.Advance(g) != read.Advance(g))
            {
                checks.Fail(property, $"glyph {g}: advance {read.Advance(g)}, the face's {face.Advance(g)}");
                return;
            }

            string? outline = CompareOutlines(face.Outline(g), read.Outline(g));
            if (outline is not null)
            {
                checks.Fail(property, $"glyph {g}: {outline}");
                return;
            }
        }

        checks.Pass(property);
    }

    private string? CompareOutlines(GlyphOutline expected, GlyphOutline actual)
    {
        if (Exact)
        {
            return expected.Verbs.Span.SequenceEqual(actual.Verbs.Span)
                ? Comparisons.Difference(expected.Points.Span, actual.Points.Span, 0f)
                : $"{actual.Verbs.Length} steps, the face's {expected.Verbs.Length}";
        }

        var (a, b) = (expected.Bounds(), actual.Bounds());
        float tolerance = _options.CurveTolerance + 1f;
        return MathF.Abs(a.MinX - b.MinX) <= tolerance && MathF.Abs(a.MinY - b.MinY) <= tolerance && MathF.Abs(a.MaxX - b.MaxX) <= tolerance && MathF.Abs(a.MaxY - b.MaxY) <= tolerance
            ? null
            : $"bounds {b}, the face's {a} (tolerance {tolerance})";
    }

    // Every probed code point answers the same glyph (for a subset: those whose glyph is kept).
    private static void CheckCharacters(IFontFace face, IFontFace read, HashSet<int>? kept, CaseChecks checks, string property)
    {
        foreach (var (firstCode, lastCode) in Probed)
        {
            for (int code = firstCode; code <= lastCode; code++)
            {
                int glyph = face.Glyph(code);
                if ((kept is null || kept.Contains(glyph)) && glyph != 0 && read.Glyph(code) != glyph)
                {
                    checks.Fail(property, $"U+{code:X4}: glyph {read.Glyph(code)}, the face's {glyph}");
                    return;
                }
            }
        }

        checks.Pass(property);
    }

    private static void CheckShaping(IFontFace face, IFontFace read, CaseChecks checks)
    {
        foreach (var (text, script, rightToLeft) in Lines)
        {
            ShapedGlyph[] expected;
            try
            {
                expected = TextShapers.Shape(face, text, new TextRun(0, text.Length, script, rightToLeft));
            }
            catch (NotRegisteredException)
            {
                checks.Skip("the same shaping", $"no text shaper is registered for {script}");
                continue;
            }

            var actual = TextShapers.Shape(read, text, new TextRun(0, text.Length, script, rightToLeft));
            checks.Compare("the same shaping", Comparisons.Sequences(expected, actual));
        }
    }

    private static JsonObject Subset(int face, int[] glyphs, bool keepLayout) =>
        new() { ["face"] = face, ["kind"] = "subset", ["glyphs"] = new JsonArray([.. glyphs.Select(g => (JsonNode)g)]), ["layout"] = keepLayout };
}
