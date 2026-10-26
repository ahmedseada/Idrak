// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Data;

// Plan 18, phase 2: text shaping, checked glyph for glyph against HarfBuzz 14 (uharfbuzz) on fixed strings in Arabic,
// Persian, Hebrew, Latin, Greek and Cyrillic (tools/fonts/make_font_references.py, tests/Idrak.Tests/data/fonts/shaping.json).
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] ShapingGroup =
    [
        ("shaping: every fixed string (Arabic joining, lam-alef, diacritics and shadda, tatweel, mirrored brackets, Persian with ZWNJ, Arabic digits, Hebrew points, Latin kerning, Greek, Cyrillic, a font without GSUB) gives HarfBuzz's glyphs, clusters, advances and offsets", ShapingMatchesHarfBuzz),
        ("shaping: real Arabic fonts with large layout tables (Amiri, Cairo and Reem Kufi, variable ones included) give HarfBuzz's glyphs, clusters, advances and offsets on 100 lines with and without harakat: marks on ligature components and on the first glyph of a multiple substitution, marks in their letter's cluster, substitution features' GPOS lookups", d => ShapingAgainst("arabic/shaping.json", TextClusters.Graphemes)),
        ("shaping: the bidirectional algorithm gives python-bidi's visual order for mixed paragraphs (Arabic with Latin, European and Arabic digits, separators, brackets, a URL, Hebrew, trailing spaces, RLM), each direction given and found", BidiMatchesReference),
        ("shaping: a paragraph is cut into runs of one script and one direction, spaces and punctuation taking their neighbours' script", ItemizerRuns),
        ("shaping: the shapers by script: arabic for the joining scripts, general for the others and for scripts with none of their own; an app's shaper replaces the library's for its script only", ShapingRegistry),
    ];

    internal static void ShapingMatchesHarfBuzz(Device device) => ShapingAgainst("shaping.json", TextClusters.Characters);   // made at HarfBuzz's character level

    // Every reference of the file (tools/fonts) shaped as HarfBuzz shapes it.
    internal static void ShapingAgainst(string referenceFile, TextClusters clusters)
    {
        var references = JsonNode.Parse(File.ReadAllText(FontData(referenceFile)))!.AsArray();
        var faces = new Dictionary<string, IFontFace>();
        int glyphs = 0;
        foreach (var reference in references)
        {
            string file = (string)reference!["font"]!, text = (string)reference["text"]!;
            if (!faces.TryGetValue(file, out var face))
            {
                faces[file] = face = FontFormats.Read(FontData(file)).Single();
            }

            var features = reference["features"] is JsonArray f ? f.Select(x => (string)x!).ToArray() : null;
            var run = new TextRun(0, text.Length, (string)reference["script"]!, (string)reference["direction"]! == "rtl", (string?)reference["language"]) { Clusters = clusters };
            var shaped = TextShapers.Shape(face, text, run, features);
            var expected = reference["glyphs"]!.AsArray().Select(g => new ShapedGlyph((int)g![0]!, (int)g[1]!, (int)g[2]!, (int)g[3]!, (int)g[4]!, (int)g[5]!)).ToArray();
            string Show(IEnumerable<ShapedGlyph> list) => string.Join(" ", list.Select(g => $"{g.Glyph}@{g.Cluster}+{g.XAdvance}({g.XOffset},{g.YOffset})"));
            Check(shaped.SequenceEqual(expected), $"{text} ({run.Script}, {(run.RightToLeft ? "rtl" : "ltr")}, {file}):\n      ours      {Show(shaped)}\n      HarfBuzz  {Show(expected)}");
            glyphs += expected.Length;
        }

        Console.WriteLine($"    {references.Count} strings, {glyphs} glyphs as HarfBuzz shapes them");
    }

    internal static void BidiMatchesReference(Device device)
    {
        var references = JsonNode.Parse(File.ReadAllText(FontData("bidi.json")))!.AsArray();
        foreach (var reference in references)
        {
            string text = (string)reference!["text"]!, expected = (string)reference["visual"]!;
            string? base_ = (string?)reference["base"];
            bool? rightToLeft = base_ switch { "L" => false, "R" => true, _ => null };
            var (levels, classes, rtl) = Idrak.Imaging.Shaping.Bidi.Resolve(text, rightToLeft);
            var (order, _) = Idrak.Imaging.Shaping.Bidi.Reorder(levels, classes, 0, text.Length, rtl);
            string visual = new([.. order.Select(i => text[i])]);
            Check(visual == expected, $"\"{text}\" ({base_ ?? "found"}): ours \"{visual}\", python-bidi \"{expected}\" (levels {string.Join("", levels)})");
        }

        Console.WriteLine($"    {references.Count} paragraphs in python-bidi's order");
    }

    internal static void ItemizerRuns(Device device)
    {
        string text = "المادة 12 من Code (2026) نظام";
        var (levels, _, rtl) = Idrak.Imaging.Shaping.Bidi.Resolve(text);
        var runs = Idrak.Imaging.Shaping.Itemizer.Runs(text, levels, 0, text.Length, "ar");
        string Show((TextRun Run, byte Level) r) => $"{text.Substring(r.Run.Start, r.Run.Length)}|{r.Run.Script}|{(r.Run.RightToLeft ? "rtl" : "ltr")}";
        var shown = runs.Select(Show).ToArray();
        Check(rtl && shown.SequenceEqual(["المادة |Arab|rtl", "12|Arab|ltr", " من |Arab|rtl", "Code (2026)|Latn|ltr", " نظام|Arab|rtl"]),
            $"runs: {string.Join(" / ", shown)}");
        Check(runs.Sum(r => r.Run.Length) == text.Length && runs.All(r => r.Run.Language == "ar"), "the runs cover the text, with its language");
    }

    internal static void ShapingRegistry(Device device)
    {
        Check(TextShapers.Get("Arab").Name == "arabic" && TextShapers.Get("syrc").Name == "arabic" && TextShapers.Get("Nkoo").Name == "arabic",
            "the joining scripts are shaped by arabic");
        Check(TextShapers.Get("Latn").Name == "general" && TextShapers.Get("Hebr").Name == "general" && TextShapers.Get("Xyzw").Name == "general",
            "other scripts, and scripts with no shaper of their own, by general");

        var app = new FixedShaper();
        TextShapers.Register("Hebr", app);
        try
        {
            var face = SubsetFace();
            Check(TextShapers.Shape(face, "שלום", new TextRun(0, 4, "Hebr", true)).Single().Glyph == 7, "the app's shaper shapes Hebrew");
            Check(TextShapers.Get("Latn").Name == "general" && TextShapers.Origin("Hebr") != Idrak.Abstraction.Overrides.Library, "and nothing else");
        }
        finally
        {
            TextShapers.Unregister("Hebr");
        }

        Check(TextShapers.Get("Hebr").Name == "general", "unregistered: general again");
    }

    internal sealed class FixedShaper : ITextShaper
    {
        public string Name => "fixed";

        public ShapedGlyph[] Shape(IFontFace font, string text, TextRun run, IReadOnlyCollection<string>? features = null) => [new ShapedGlyph(7, run.Start, 100, 0, 0, 0)];
    }
}
