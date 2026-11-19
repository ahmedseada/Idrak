// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text.Json.Nodes;

// Plan 28, phase D10 (decision 17): idrak data redact on the D10 redaction sample tools/documents/make_fixtures.py writes
// into the ignored test-output/documents/samples, and on a file of many pages made here. The redacted files are kept in
// test-output/documents/redacted for tools/documents/check_written.py.
internal abstract partial class DesktopTests
{
    internal static readonly (string Name, Action<Device> Run)[] CliRedactGroup =
    [
        ("cli: data redact: the redaction sample's own marks (--marked), its words (--text, again for more) and boxes (--boxes) removed for real: the words gone from the written file's text, the drawing inside and the image pixels under the area, the marks applied; a line per page says what was removed, --json the same; a file of 500 pages through the command (its time)", CliRedactSample),
        ("cli: data redact: usage errors: no areas, no output, the output the input, a missing file, a boxes file that is not areas, a page past the end, a colour that is not one", CliRedactUsage),
        ("cli: data redact: a note under a mark leaves with its popup, its replies (one on the next page), the fields whose widgets lie under a mark and what names them (--marked on redaction-replies): no removed word in the file, note-5 alone left", CliRedactReplies),
        ("cli: data redact: words split across text operators, fonts and kerned pieces found whole and removed (--text with --ignore-case on redaction-split): two areas a word, none of their pieces left in any decoded stream", CliRedactSplit),
        ("cli: data redact: images in a pattern, a soft mask's group and a Type 3 glyph, and an image's soft mask, redacted (--marked on redaction-nested and redaction-masked): their pixels as their .redacted.png say, glyph B's description and the images as they were gone from the file", CliRedactImages),
    ];

    private static (int Code, string Out, string Err) RedactCli(params string[] args)
    {
        var output = new StringWriter();
        var error = new StringWriter();
        int code = Idrak.Cli.CommandLine.Run([.. args, "-C", Path.Combine(Path.GetTempPath(), "idrak-cli-redact-no-config.json")], output, error);
        return (code, output.ToString(), error.ToString());
    }

    // The D10 redaction sample (IDRAK_DOCUMENTS_SAMPLES, else test-output/documents/samples): missing, the group fails naming the command that makes it.
    private static string RedactSample(string name)
    {
        string folder = Environment.GetEnvironmentVariable("IDRAK_DOCUMENTS_SAMPLES") is { Length: > 0 } set ? set : RepositoryPath("test-output", "documents", "samples");
        string path = Path.Combine(folder, "d10", name + ".pdf");
        Check(File.Exists(path), $"{path} is not there. Make the samples with: python -I tools/documents/make_fixtures.py -o test-output/documents/samples");
        return path;
    }

    private static string RedactFolder() => Directory.CreateDirectory(RepositoryPath("test-output", "documents", "redacted")).FullName;

    private static string RedactText(string path)
    {
        using var document = DocumentFormats.Open(path);
        return string.Join("\n", Enumerable.Range(0, document.PageCount).Select(i => document.ReadPage(i, DocumentParts.Text).Text));
    }

    private static void CliRedactSample(Device device)
    {
        CpuOnly(device);
        string sample = RedactSample("redaction"), folder = RedactFolder();
        string marked = Path.Combine(folder, "redaction-cli-marked.pdf");
        var (code, output, error) = RedactCli("data", "redact", sample, "--marked", "-o", marked);
        Check(code == 0, $"data redact --marked: exit {code}: {error}{output}");
        Check(output.Contains("page 1: 3 areas: 16 glyphs (16 characters), 1 drawings removed, 1 clipped, 1 images", StringComparison.Ordinal)
              && output.Contains("3 annotations", StringComparison.Ordinal) && output.Contains($"written to {marked}", StringComparison.Ordinal), $"the summary:\n{output}");
        string text = RedactText(marked);
        Check(!text.Contains("Robin Park", StringComparison.Ordinal) && !text.Contains("48,000", StringComparison.Ordinal) && text.Contains("Meeting room booked", StringComparison.Ordinal),
            $"the text redacted: {text}");
        using (var document = DocumentFormats.Open(marked))
        {
            var page = document.ReadPage(0, DocumentParts.All);
            Check(page.Annotations.Count == 0 && page.Images.Count == 1 && page.Paths.Count >= 5, $"marks applied ({page.Annotations.Count} left), the image and paths drawn ({page.Images.Count}, {page.Paths.Count})");
        }

        File.WriteAllText(Path.ChangeExtension(marked, ".json"), RedactJson(marked));
        var (jsonCode, json, _) = RedactCli("data", "redact", sample, "--marked", "-o", Path.Combine(folder, "redaction-cli-json.pdf"), "--json");
        var report = JsonNode.Parse(json)!;                                    // --json: the report alone, indented
        Check(jsonCode == 0 && report["pages"]![0]!["glyphs"]!.GetValue<int>() == 16 && report["pages"]![0]!["pathsRemoved"]!.GetValue<int>() == 1
              && report["pages"]![0]!["pixels"]!.GetValue<long>() > 0, $"--json: {json}");

        string texts = Path.Combine(folder, "redaction-cli-text.pdf");
        (code, output, error) = RedactCli("data", "redact", sample, "--text", "Robin Park", "--text", "48,000", "--fill", "#336699", "-o", texts);
        Check(code == 0 && output.Contains("page 1: 2 areas: 16 glyphs", StringComparison.Ordinal) && !RedactText(texts).Contains("Robin", StringComparison.Ordinal),
            $"data redact --text: exit {code}: {error}{output}");
        File.WriteAllText(Path.ChangeExtension(texts, ".json"), RedactJson(texts));

        string areas = Path.Combine(folder, "redaction-areas.json"), boxes = Path.Combine(folder, "redaction-cli-boxes.pdf");
        File.WriteAllText(areas, """[{"page": 1, "box": [25, 90, 300, 14], "fill": "none"}]""");
        (code, output, error) = RedactCli("data", "redact", sample, "--boxes", areas, "-o", boxes);
        Check(code == 0 && !RedactText(boxes).Contains("Meeting room", StringComparison.Ordinal) && RedactText(boxes).Contains("Robin Park", StringComparison.Ordinal),
            $"data redact --boxes: exit {code}: {error}{output}");
        File.WriteAllText(Path.ChangeExtension(boxes, ".json"), RedactJson(boxes));

        // A file of many pages through the command: each page's account number removed.
        var face = SubsetFace();
        string big = Path.Combine(folder, "pages-500.pdf"), bigRedacted = Path.Combine(folder, "pages-500-redacted.pdf");
        using (var builder = DocumentWriters.Create(big))
        {
            for (int i = 0; i < 500; i++)
            {
                var page = builder.AddPage(595, 842);
                foreach (var (line, y) in new[] { (string.Create(CultureInfo.InvariantCulture, $"Report page {i + 1}: account 4711-{i:D4} closed."), 100f), ("The quarterly figures follow below.", 130f) })
                {
                    var glyphs = new List<DocumentGlyph>();
                    float x = 60, scale = 12f / face.Metrics.UnitsPerEm;
                    foreach (char c in line)
                    {
                        int g = face.Glyph(c);
                        glyphs.Add(new DocumentGlyph(g, x, y, face.Advance(g) * scale));
                        x += face.Advance(g) * scale;
                    }

                    page.Glyphs(face, 12, [.. glyphs], line, DocumentTextMode.Fill, DocumentColour.Black);
                }
            }

            builder.Finish();
        }

        long start = Stopwatch.GetTimestamp();
        (code, output, error) = RedactCli("data", "redact", big, "--text", "4711-", "-o", bigRedacted);
        var elapsed = Stopwatch.GetElapsedTime(start);
        Check(code == 0 && output.Contains("500 areas on 500 pages", StringComparison.Ordinal), $"500 pages: exit {code}: {error}{output[..Math.Min(output.Length, 400)]}");
        string redacted = RedactText(bigRedacted);
        Check(!redacted.Contains("4711", StringComparison.Ordinal) && redacted.Contains("Report page 500", StringComparison.Ordinal), "the account numbers gone from every page");
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"    500 pages ({new FileInfo(big).Length:N0} bytes) redacted through the command: {elapsed.TotalSeconds:F2} s ({elapsed.TotalMilliseconds / 500:F2} ms a page), {new FileInfo(bigRedacted).Length:N0} bytes written"));
    }

    // A written file's text, page by page, for the checker; with `removed`, the words no tool may find in it.
    private static string RedactJson(string path, IEnumerable<string>? removed = null)
    {
        using var document = DocumentFormats.Open(path);
        var pages = new JsonArray();
        for (int i = 0; i < document.PageCount; i++)
        {
            var page = document.ReadPage(i, DocumentParts.Text);
            pages.Add(new JsonObject { ["index"] = i, ["width"] = page.Width, ["height"] = page.Height, ["text"] = page.Text });
        }

        string name = Path.GetFileNameWithoutExtension(path);
        var json = new JsonObject { ["schema"] = 1, ["name"] = name, ["file"] = name + ".pdf", ["pages"] = pages };
        if (removed?.ToList() is { Count: > 0 } words)
        {
            json["removedText"] = new JsonArray([.. words.Select(w => (JsonNode)w)]);
        }

        return json.ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = true }) + "\n";
    }

    // A redaction sample's JSON: what applying its redaction must leave.
    private static JsonObject RedactExpected(string name) => JsonNode.Parse(File.ReadAllText(Path.ChangeExtension(RedactSample(name), ".json")))!["redaction"]!.AsObject();

    private static IEnumerable<string> RedactStrings(JsonNode? node) => node is JsonArray array ? array.Select(v => (string)v!) : [];

    private static void CliRedactReplies(Device device)
    {
        CpuOnly(device);
        string sample = RedactSample("redaction-replies"), output = Path.Combine(RedactFolder(), "redaction-replies-cli.pdf");
        var after = RedactExpected("redaction-replies")["afterApplying"]!.AsObject();
        var words = RedactStrings(after["removedText"]).ToList();
        var (code, text, error) = RedactCli("data", "redact", sample, "--marked", "-o", output);
        Check(code == 0 && text.Contains("page 1: 2 areas: 16 glyphs (16 characters)", StringComparison.Ordinal) && text.Contains(", 8 annotations", StringComparison.Ordinal),
            $"data redact --marked: exit {code}: {error}{text}");
        var left = PdfLeftovers(File.ReadAllBytes(output), words);
        Check(left.Count == 0, $"removed words still in the written file: {string.Join("; ", left)}");
        using (var document = DocumentFormats.Open(output))
        {
            Check(document.ReadPage(0, DocumentParts.Annotations).Annotations.Select(a => a.Id).SequenceEqual(RedactStrings(after["keptAnnotations"]))
                  && document.ReadPage(1, DocumentParts.Annotations).Annotations.Count == 0 && document.Fields.Count == 0, "note-5 alone left, no field");
        }

        File.WriteAllText(Path.ChangeExtension(output, ".json"), RedactJson(output, words));
    }

    private static void CliRedactSplit(Device device)
    {
        CpuOnly(device);
        string sample = RedactSample("redaction-split"), output = Path.Combine(RedactFolder(), "redaction-split-cli.pdf");
        var redaction = RedactExpected("redaction-split");
        var words = RedactStrings(redaction["words"]).ToList();
        var pieces = RedactStrings(redaction["afterApplying"]!["removedPieces"]).ToList();
        string[] args = ["data", "redact", sample, .. words.SelectMany((w, k) => new[] { "--text", k % 2 == 0 ? w.ToUpperInvariant() : w.ToLowerInvariant() }), "--ignore-case", "-o", output];
        var (code, text, error) = RedactCli(args);
        int glyphs = words.Sum(w => w.Length);
        Check(code == 0 && text.Contains($"page 1: {2 * words.Count} areas: {glyphs} glyphs ({glyphs} characters)", StringComparison.Ordinal), $"data redact --text: exit {code}: {error}{text}");
        var left = PdfLeftovers(File.ReadAllBytes(output), pieces);
        Check(left.Count == 0, $"pieces of the removed words still in the written file: {string.Join("; ", left)}");
        string written = RedactText(output);
        Check(!words.Any(w => written.Contains(w, StringComparison.OrdinalIgnoreCase)) && written.Contains("Nothing else changes here", StringComparison.Ordinal), $"the text: {written}");
        File.WriteAllText(Path.ChangeExtension(output, ".json"), RedactJson(output, [.. words, .. pieces]));
    }

    private static void CliRedactImages(Device device)
    {
        CpuOnly(device);
        string nested = RedactSample("redaction-nested"), output = Path.Combine(RedactFolder(), "redaction-nested-cli.pdf");
        var after = RedactExpected("redaction-nested")["afterApplying"]!.AsObject();
        var (code, text, error) = RedactCli("data", "redact", nested, "--marked", "-o", output);
        Check(code == 0 && text.Contains("page 1: 3 areas: 1 glyphs (1 characters), 0 drawings removed, 2 clipped, 3 images", StringComparison.Ordinal), $"data redact --marked: exit {code}: {error}{text}");
        byte[] bytes = File.ReadAllBytes(output);
        var gone = RedactStrings(after["removedBytes"]).Select(h => ("glyph B's image", Convert.FromHexString(h))).ToList();
        var originals = new[] { (string)after["patternImage"]!["original"]!, (string)after["maskImage"]!["original"]! };
        Check(PdfLeftoverBytes(bytes, gone).Count == 0 && !PdfImageObjects(bytes).Any(o => originals.Contains(o.Pixels)), "glyph B's description and the images as they were gone from the file");
        using (var document = DocumentFormats.Open(output))
        {
            var page = document.ReadPage(0, DocumentParts.All);
            Check(PdfPixelHash(page.Paths.Single(p => p.FillPattern?.Cell is not null).FillPattern!.Cell!.Images.Single().Decode()) == (string)after["patternImage"]!["sha256"]!
                  && PdfPixelHash(page.SoftMasks.Single().Content.Images.Single().Decode()) == (string)after["maskImage"]!["sha256"]!, "the pattern's and the mask's images as their .redacted.png say");
        }

        File.WriteAllText(Path.ChangeExtension(output, ".json"), RedactJson(output));

        string masked = RedactSample("redaction-masked"), maskedOutput = Path.Combine(RedactFolder(), "redaction-masked-cli.pdf");
        var expected = RedactExpected("redaction-masked")["afterApplying"]!.AsObject();
        (code, text, error) = RedactCli("data", "redact", masked, "--marked", "-o", maskedOutput);
        Check(code == 0 && text.Contains("1 images", StringComparison.Ordinal), $"data redact --marked: exit {code}: {error}{text}");
        using (var document = DocumentFormats.Open(maskedOutput))
        {
            var picture = document.ReadPage(0, DocumentParts.Images).Images.Single();
            Check(PdfPixelHash(picture.Decode()) == (string)expected["image"]!["sha256"]! && picture.Mask() is { } grey && PdfPixelHash(grey) == (string)expected["mask"]!["sha256"]!,
                "the image and its kept soft mask as their .redacted.png say");
        }

        File.WriteAllText(Path.ChangeExtension(maskedOutput, ".json"), RedactJson(maskedOutput));
    }

    private static void CliRedactUsage(Device device)
    {
        CpuOnly(device);
        string sample = RedactSample("redaction"), folder = RedactFolder();
        string output = Path.Combine(folder, "usage.pdf");
        void Refused(string why, string text, params string[] args)
        {
            var (code, _, error) = RedactCli(["data", "redact", .. args]);
            Check(code == 2 && error.Contains(text, StringComparison.Ordinal), $"{why}: exit {code}: {error}");
        }

        Refused("no areas", "--boxes FILE, --text TEXT or --marked", sample, "-o", output);
        Refused("no output", "-o, --out", sample, "--marked");
        Refused("the output the input", "another file", sample, "--marked", "-o", sample);
        Refused("a missing file", "is not there", Path.Combine(folder, "nope.pdf"), "--marked", "-o", output);
        string bad = Path.Combine(folder, "bad-areas.json");
        File.WriteAllText(bad, """{"page": 1}""");
        Refused("a boxes file that is not an array", "JSON array", sample, "--boxes", bad, "-o", output);
        File.WriteAllText(bad, """[{"page": 1, "box": [1, 2, 3]}]""");
        Refused("an area without four numbers", "area 1", sample, "--boxes", bad, "-o", output);
        File.WriteAllText(bad, """[{"page": 9, "box": [1, 2, 3, 4]}]""");
        Refused("a page past the end", "page 9", sample, "--boxes", bad, "-o", output);
        Refused("a colour that is not one", "not a colour", sample, "--marked", "--fill", "teal-ish", "-o", output);
        Check(!File.Exists(output) && !File.Exists(output + ".partial"), "nothing written when refused");
    }
}
