// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Imaging;
using Idrak.Text;
using Idrak.Text.Abstractions;

namespace Idrak.Cli.Commands.Data;

/// <summary>
/// <c>idrak data fonts DIR [--harfbuzz REFS.json]</c>: whether fonts can draw a language's pages (plan 18): each font
/// read, the sample's characters it lacks, its layout features for the script, and, given HarfBuzz's shaping of the same
/// lines (tools/fonts/harfbuzz_reference.py), whether idrak shapes every line as HarfBuzz does.
/// </summary>
internal sealed class DataFontsCommand : Command
{
    public override string Name => "data fonts";

    public override string Summary => "Check fonts for drawing pages: characters, layout features, and shaping against HarfBuzz";

    public override string Usage => """
        FONTS... [--language CODE] [--script NAME] [--lines FILE] [--write-lines OUT] [--harfbuzz JSON]

        Reads every font (.ttf, .otf, .ttc, folders searched through) and says for each: whether it reads, which
        characters of the sample it has no glyph for (a page with one of them is left out), its layout features for the
        script, and for a joining script (Arabic) whether it has the joining forms (init, medi, fina), the
        lam-alef ligatures (rlig) and mark positioning (mark). With --harfbuzz, every line of the references is shaped
        here too and compared glyph for glyph (glyphs, clusters, advances, offsets).

        Options:
          -l, --language CODE   the sample's language: random text of it unless --lines (ar)
              --script SCRIPT   the script to check, ISO 15924 (Arab)
              --lines FILE      the sample: a text file, a line a shaping (blank lines and --- passed over)
              --write-lines F   write the sample's lines, for tools/fonts/harfbuzz_reference.py
              --harfbuzz FILE   HarfBuzz's shaping of the lines in the fonts (that tool's JSON)

        Comparing with HarfBuzz takes three steps (Python with pip install uharfbuzz for the second):
          idrak data fonts DIR --write-lines lines.txt
          python tools/fonts/harfbuzz_reference.py DIR lines.txt -o hb.json
          idrak data fonts DIR --harfbuzz hb.json

        Examples:
          idrak data fonts fonts
          idrak data fonts fonts\Amiri\Amiri-Regular.ttf --lines my-lines.txt
          idrak data fonts fonts -l en --script Latn
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } = ["--language", "--script", "--lines", "--write-lines", "--harfbuzz"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-l"] = "--language" };

    public override int Run(CommandContext context)
    {
        if (context.Positional.Count == 0)
        {
            throw new UsageException("Give the fonts: a folder or font files.");
        }

        string language = context.Option("--language") ?? "ar";
        string script = context.Option("--script") ?? "Arab";
        List<string> lines;
        try
        {
            lines = context.Option("--lines") is { } file
                ? [.. File.ReadLines(file, Encoding.UTF8).Select(l => l.Trim()).Where(l => l.Length > 0 && l != "---")]
                : [.. RandomText.Documents(RandomTextSources.Get(language), new RandomTextOptions { Unit = RandomTextUnit.Lines, Count = 200, Diacritics = 0.3, LineWords = (3, 10) })];
        }
        catch (Exception e) when (e is IOException or NotSupportedException or ArgumentException)
        {
            throw new UsageException(e.Message);
        }

        if (lines.Count == 0)
        {
            throw new UsageException("The sample has no lines.");
        }

        if (context.Option("--write-lines") is { } written)
        {
            File.WriteAllText(written, string.Join("\n", lines) + "\n", new UTF8Encoding(false));
            context.Write(string.Create(CultureInfo.InvariantCulture, $"{lines.Count:N0} lines written to {written}"));
        }

        var references = context.Option("--harfbuzz") is { } hb ? Read(hb) : null;
        var needed = new SortedSet<int>(lines.SelectMany(l => l.EnumerateRunes()).Where(r => !Rune.IsWhiteSpace(r) && !Rune.IsControl(r)
            && Rune.GetUnicodeCategory(r) != UnicodeCategory.Format).Select(r => r.Value));
        bool joining = TextShapers.Get(script).Name == "arabic";

        var reports = new JsonArray();
        int problems = 0;
        foreach (var (root, path) in context.Positional.SelectMany(Files))
        {
            string relative = Path.GetRelativePath(root, path).Replace('\\', '/');
            IReadOnlyList<IFontFace> faces;
            try
            {
                faces = FontFormats.Read(path);
            }
            catch (Exception e) when (e is InvalidDataException or NotSupportedException or IOException)
            {
                problems++;
                context.Write($"{relative}: not read: {e.Message}");
                reports.Add(item: new JsonObject { ["font"] = relative, ["error"] = e.Message });
                continue;
            }

            for (int index = 0; index < faces.Count; index++)
            {
                var face = faces[index];
                var missing = needed.Where(cp => face.Glyph(cp) == 0).ToList();
                var (substitution, positioning) = OpenTypeFeatures.Of(face, script, language);
                string[] forms = ["init", "medi", "fina"];   // isolated letters are the plain glyphs: isol is often absent
                var lacking = !joining ? [] : forms.Where(f => !substitution.Contains(f)).Concat(substitution.Contains("rlig") ? [] : ["rlig"])
                    .Concat(positioning.Contains("mark") ? [] : ["mark"]).ToList();
                var report = new JsonObject
                {
                    ["font"] = relative, ["face"] = index, ["family"] = face.Family, ["style"] = face.Style, ["glyphs"] = face.GlyphCount,
                    ["missing"] = new JsonArray([.. missing.Select(cp => (JsonNode)$"U+{cp:X4} {char.ConvertFromUtf32(cp)}")]),
                    ["substitution"] = new JsonArray([.. substitution.Select(t => (JsonNode)t)]),
                    ["positioning"] = new JsonArray([.. positioning.Select(t => (JsonNode)t)]),
                    ["lacking"] = new JsonArray([.. lacking.Select(t => (JsonNode)t)]),
                };

                string line = $"{relative}{(faces.Count > 1 ? $" #{index}" : "")}: {face.Family} {face.Style}, {face.GlyphCount:N0} glyphs";
                line += missing.Count == 0 ? $"; has every character of the sample ({needed.Count})"
                    : $"; lacks {missing.Count} of {needed.Count} characters: {string.Join(" ", missing.Take(12).Select(cp => $"U+{cp:X4} {char.ConvertFromUtf32(cp)}"))}{(missing.Count > 12 ? " ..." : "")}";
                if (joining)
                {
                    line += lacking.Count == 0 ? "; joining forms, rlig and mark: all there" : $"; no {string.Join(", ", lacking)} for {script}";
                }

                if (references is not null)
                {
                    var (same, total, first) = Compare(face, relative, index, references);
                    report["harfbuzz"] = new JsonObject { ["identical"] = same, ["lines"] = total, ["first"] = first };
                    line += total == 0 ? "; no HarfBuzz references for it" : $"; HarfBuzz: {same} of {total} lines identical";
                    if (same < total)
                    {
                        problems++;
                        line += $"\n    first: {first}";
                    }
                }

                problems += missing.Count > 0 || lacking.Count > 0 ? 1 : 0;
                context.Write(line);
                reports.Add(item: report);
            }
        }

        if (reports.Count == 0)
        {
            throw new UsageException($"No font (.ttf, .otf, .ttc) in {string.Join(", ", context.Positional)}.");
        }

        if (references is null && !context.Json)
        {
            context.Write("to compare with HarfBuzz: --write-lines lines.txt, python tools/fonts/harfbuzz_reference.py DIR lines.txt -o hb.json, then --harfbuzz hb.json");
        }

        context.WriteJson(new JsonObject { ["script"] = script, ["language"] = language, ["lines"] = lines.Count, ["characters"] = needed.Count, ["fonts"] = reports });
        return problems == 0 ? ExitCodes.Ok : ExitCodes.Failed;
    }

    private static IEnumerable<(string Root, string Path)> Files(string target)
    {
        if (File.Exists(target))
        {
            return [(Path.GetDirectoryName(Path.GetFullPath(target))!, Path.GetFullPath(target))];
        }

        if (!Directory.Exists(target))
        {
            throw new UsageException($"{target} is neither a font file nor a folder.");
        }

        string root = Path.GetFullPath(target);
        return Directory.EnumerateFiles(root, "*", SearchOption.AllDirectories)
            .Where(f => Path.GetExtension(f).ToLowerInvariant() is ".ttf" or ".otf" or ".ttc")
            .Order(StringComparer.Ordinal).Select(f => (root, f));
    }

    private static JsonArray Read(string path)
    {
        try
        {
            return JsonNode.Parse(File.ReadAllBytes(path)) as JsonArray ?? throw new UsageException($"{path} is not a list of HarfBuzz references.");
        }
        catch (Exception e) when (e is IOException or System.Text.Json.JsonException)
        {
            throw new UsageException($"{path}: {e.Message}");
        }
    }

    // The references for this font (by its path under the folder, or its file name), each shaped here and compared.
    private static (int Same, int Total, string? First) Compare(IFontFace face, string relative, int index, JsonArray references)
    {
        int same = 0, total = 0;
        string? first = null;
        foreach (var reference in references)
        {
            string font = (string?)reference?["font"] ?? "";
            if ((font != relative && Path.GetFileName(font) != Path.GetFileName(relative)) || ((int?)reference!["face"] ?? 0) != index)
            {
                continue;
            }

            string text = (string)reference["text"]!;
            var run = new TextRun(0, text.Length, (string?)reference["script"] ?? "Arab", (string?)reference["direction"] != "ltr", (string?)reference["language"])
            {
                Clusters = (string?)reference["clusters"] == "characters" ? TextClusters.Characters : TextClusters.Graphemes,
            };
            var ours = TextShapers.Shape(face, text, run);
            var theirs = reference["glyphs"]!.AsArray().Select(g => new ShapedGlyph((int)g![0]!, (int)g[1]!, (int)g[2]!, (int)g[3]!, (int)g[4]!, (int)g[5]!)).ToArray();
            total++;
            if (ours.SequenceEqual(theirs))
            {
                same++;
                continue;
            }

            if (first is null)
            {
                int at = 0;
                while (at < Math.Min(ours.Length, theirs.Length) && ours[at] == theirs[at])
                {
                    at++;
                }

                static string Show(ShapedGlyph[] g, int at) => at < g.Length ? $"{g[at].Glyph}@{g[at].Cluster}+{g[at].XAdvance}({g[at].XOffset},{g[at].YOffset})" : "nothing";
                first = $"\"{text}\" at glyph {at}: idrak {Show(ours, at)}, HarfBuzz {Show(theirs, at)} ({ours.Length} and {theirs.Length} glyphs)";
            }
        }

        return (same, total, first);
    }
}
