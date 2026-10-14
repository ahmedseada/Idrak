// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.IO.Compression;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Data;
using Idrak.Abstraction.Generation;
using Idrak.Nlp;
using Idrak.Nlp.Abstractions;

// Plan 18, phase 5: tuning data from text: records' answers drawn as pages, scanned, written again with the page and the
// drawn text, read back by the tuning data formats.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] SyntheticDataGroup =
    [
        ("synthetic data: ShareGPT records with JSON answers become pages: each record written again in its shape with one page image, its answer's full text the drawn text, its other fields, fence and indentation kept, a manifest line a page; the tuning reader reads them with their images", SyntheticShareGpt),
        ("synthetic data: the same seed gives the same bytes on one worker or many, another seed other pages; images in a zip read back with the zip as the images root", SyntheticRepeatable),
        ("synthetic data: plain text documents and chat messages records are read, and written with one image each", SyntheticFormats),
        ("synthetic data: page parts fill their slots (digits, number, pick, field, date, hijri, line), drop a line whose field is missing, use the language's digits when drawn, keep a part out at chance 0; unknown keys and slots are refused", SyntheticParts),
        ("synthetic data: a record no font can draw, or with no text, is left out and said why; a font without a page's letters is not used for it; every record left out is an error", SyntheticSkips),
        ("synthetic data: the record formats and slots by name; an app's slot replaces the library's of its name only", SyntheticRegistries),
    ];

    private static string SyntheticFolder(string name)
    {
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-synthetic-{name}-{Guid.NewGuid():N}");
        Directory.CreateDirectory(folder);
        return folder;
    }

    private const string PartsJson = """
        {
          "headerStart": [["المملكة العربية السعودية", "وزارة العدل"], ["{pick:المحكمة العامة|المحكمة الجزائية}"]],
          "headerEnd": [["الرقم: {digits:5}", "التاريخ: {hijri} هـ"]],
          "subjects": ["الموضوع: {field:subject}"],
          "signatures": [["مدير الإدارة"]],
          "footers": ["هاتف {number:1000-9999}"],
          "chances": {"headerStart": 1, "headerEnd": 1, "subject": 1, "signature": 1, "footer": 1},
          "language": "ar"
        }
        """;

    private static SyntheticPageOptions SyntheticOptions(string folder, int variants = 2, int workers = 2, int seed = 7) => new()
    {
        Style = new PageStyle { Fonts = [SubsetFace(), FontFormats.Read(FontData("features.ttf")).Single()], Width = 620, Height = 877, BodySize = (14, 18) },
        Parts = SyntheticPageParts.FromJson(JsonNode.Parse(PartsJson)!.AsObject(), folder),
        Variants = variants, Workers = workers, Seed = seed, Scans = ["office", "none"],
    };

    private static string ShareGptInput(string folder)
    {
        string Record(int i, string answer) => SyntheticPages.Json(new JsonObject
        {
            ["conversations"] = new JsonArray(
                new JsonObject { ["from"] = "human", ["value"] = "<image>Extract details to JSON." },
                new JsonObject { ["from"] = "gpt", ["value"] = answer }),
            ["images"] = new JsonArray($"pages/{i}.png"),
            ["id"] = $"r{i}",
        });

        string fenced = "```json\n" + SyntheticPages.Json(new JsonObject { ["subject"] = "طلب نسخة", ["document_number"] = "4521", ["full_text_extract"] = "سعادة رئيس المحكمة المحترم\nنفيدكم بأن المدعي تقدم بطلب نسخة من الحكم رقم 4521." }) + "\n```";
        string indented = SyntheticPages.Json(new JsonObject { ["full_text_extract"] = "قرار رقم 77 لعام 1446\nيعتمد ما يلي.", ["date"] = "1446/05/02" }, indented: true);
        string plain = SyntheticPages.Json(new JsonObject { ["full_text_extract"] = "The court met on 3 May 2025 and heard the claim.", ["court"] = "General" });
        string path = Path.Combine(folder, "train.jsonl");
        File.WriteAllLines(path, [Record(0, fenced), Record(1, indented), Record(2, plain)], new UTF8Encoding(false));
        return path;
    }

    private static void SyntheticShareGpt(Device device)
    {
        string folder = SyntheticFolder("sharegpt");
        try
        {
            string input = ShareGptInput(folder);
            var reports = new List<SyntheticPagesProgress>();
            var result = SyntheticPages.Write(input, Path.Combine(folder, "out", "synthetic.jsonl"), SyntheticOptions(folder), new SyncProgress<SyntheticPagesProgress>(reports.Add));
            Check(result.Records == 3 && result.Pages == 6 && result.Skips.Count == 0, $"{result.Records} records, {result.Pages} pages, {result.Skips.Count} left out");
            Check(reports.Count > 0 && reports[^1].Pages == 6 && reports[^1].Total == 6, $"progress to {reports.LastOrDefault()?.Pages}/{reports.LastOrDefault()?.Total}");

            var lines = File.ReadAllLines(result.Data);
            var manifest = File.ReadAllLines(result.Manifest).Select(l => JsonNode.Parse(l)!).ToArray();
            Check(lines.Length == 6 && manifest.Length == 6, $"{lines.Length} records, {manifest.Length} manifest lines");
            for (int i = 0; i < lines.Length; i++)
            {
                var record = JsonNode.Parse(lines[i])!.AsObject();
                var turns = record["conversations"]!.AsArray();
                string image = (string)record["images"]![0]!;
                Check(record["images"]!.AsArray().Count == 1 && image == (string)manifest[i]["image"]! && File.Exists(Path.Combine(folder, "out", image)), $"record {i}: one image, {image}");
                Check((string)turns[0]!["value"]! == "<image>Extract details to JSON." && (string)record["id"]! == $"r{i / 2}", $"record {i}: its question and id kept");
                string answer = (string)turns[1]!["value"]!;
                string json = answer.StartsWith("```json\n", StringComparison.Ordinal) ? answer["```json\n".Length..^"\n```".Length] : answer;
                var fields = JsonNode.Parse(json)!.AsObject();
                string text = (string)fields["full_text_extract"]!;
                Check(text.Contains((i / 2) switch { 0 => "سعادة رئيس المحكمة المحترم", 1 => "قرار رقم 77 لعام 1446", _ => "The court met on 3 May 2025" }, StringComparison.Ordinal),
                    $"record {i}: the body drawn and in the answer");
                Check(text.Contains("مدير الإدارة", StringComparison.Ordinal) && text.Contains("الرقم: ", StringComparison.Ordinal) && text.Contains("هاتف ", StringComparison.Ordinal),
                    $"record {i}: the header, signature and footer drawn and in the answer");
                switch (i / 2)
                {
                    case 0:
                        Check(answer.StartsWith("```json\n", StringComparison.Ordinal) && (string)fields["document_number"]! == "4521" && text.Contains("الموضوع: طلب نسخة", StringComparison.Ordinal),
                            "the fenced answer stays fenced, its fields kept, its subject drawn from its field");
                        break;
                    case 1:
                        Check(json.Contains('\n') && (string)fields["date"]! == "1446/05/02" && !text.Contains("الموضوع", StringComparison.Ordinal), "the indented answer stays indented; no subject field, no subject line");
                        break;
                    default:
                        Check(!json.Contains('\n') && fields.Select(f => f.Key).SequenceEqual(["full_text_extract", "court"]), "the one-line answer stays one line, its keys in order");
                        break;
                }

                Check(((string)manifest[i]["page"]!).Contains("DejaVu", StringComparison.Ordinal), $"record {i}: drawn in the font with Arabic, not in features.ttf ({manifest[i]["page"]})");
            }

            var transcripts = TuningDataFormats.Read(result.Data).ToList();
            Check(transcripts.Count == 6, $"the tuning reader reads {transcripts.Count} records");
            foreach (var transcript in transcripts)
            {
                var images = transcript.Messages.SelectMany(m => m.Parts).OfType<ChatImage>().ToList();
                var decoded = ImageCodecs.Get("jpeg").Decode(images.Single().Data.Span);
                Check(decoded.Width == 620 && decoded.Height >= 877, $"its page decodes, {decoded.Width}x{decoded.Height}");
            }
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void SyntheticRepeatable(Device device)
    {
        string folder = SyntheticFolder("repeat");
        try
        {
            string input = ShareGptInput(folder);
            string Run(string name, int workers, int seed, bool zip = false)
            {
                var options = SyntheticOptions(folder, workers: workers, seed: seed) with { Zip = zip };
                return SyntheticPages.Write(input, Path.Combine(folder, name, "pages.jsonl"), options).Data;
            }

            string one = Run("one", 1, 7), many = Run("many", 4, 7), other = Run("other", 4, 8);
            Check(File.ReadAllText(one) == File.ReadAllText(many), "the records the same on one worker and four");
            var names = Directory.GetFiles(Path.Combine(folder, "one", "pages")).Select(Path.GetFileName).Order(StringComparer.Ordinal).ToArray();
            Check(names.Length == 6 && names.All(n => File.ReadAllBytes(Path.Combine(folder, "one", "pages", n!)).SequenceEqual(File.ReadAllBytes(Path.Combine(folder, "many", "pages", n!)))),
                "every image the same bytes");
            Check(File.ReadAllText(one) != File.ReadAllText(other), "another seed, other pages");

            string zipped = Run("zipped", 2, 7, zip: true);
            string zipPath = Path.ChangeExtension(zipped, ".zip");
            using (var archive = ZipFile.OpenRead(zipPath))
            {
                Check(archive.Entries.Count == 6 && archive.Entries.All(e => e.FullName.StartsWith("pages/", StringComparison.Ordinal)), $"{archive.Entries.Count} pages in the zip");
            }

            Check(!Directory.Exists(Path.Combine(folder, "zipped", "pages")), "no image folder beside a zip");
            var transcripts = TuningDataFormats.Read(zipped, options: new TuningDataOptions { Images = zipPath }).ToList();
            Check(transcripts.Count == 6 && transcripts.All(t => t.Messages.SelectMany(m => m.Parts).OfType<ChatImage>().Count() == 1), "read back with the zip as the images root");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void SyntheticFormats(Device device)
    {
        string folder = SyntheticFolder("formats");
        try
        {
            string text = Path.Combine(folder, "letters.txt");
            File.WriteAllText(text, "الحمد لله وحده\nوبعد فهذا قرار.\n---\n\n---\nSecond document, in English.\n", new UTF8Encoding(false));
            Check(SyntheticRecordFormats.Detect(text).Name == "text", "a .txt is plain text");
            var options = SyntheticOptions(folder, variants: 1) with { Parts = SyntheticPageParts.Empty, AnswerKey = "text", Prompt = "Read the page." };
            var result = SyntheticPages.Write(text, Path.Combine(folder, "from-text.jsonl"), options);
            Check(result.Records == 2 && result.Pages == 2, $"two documents (the empty one passed over): {result.Records} records, {result.Pages} pages");
            var first = JsonNode.Parse(File.ReadLines(result.Data).First())!;
            Check((string)first["conversations"]![0]!["value"]! == "<image>Read the page." && JsonNode.Parse((string)first["conversations"]![1]!["value"]!)!["text"]!.ToString() == "الحمد لله وحده\nوبعد فهذا قرار.",
                "written as ShareGPT, the prompt asked, the drawn text under the answer key");

            string chat = Path.Combine(folder, "chat.jsonl");
            File.WriteAllText(chat, SyntheticPages.Json(new JsonObject
            {
                ["messages"] = new JsonArray(
                    new JsonObject { ["role"] = "user", ["content"] = new JsonArray(new JsonObject { ["type"] = "image", ["path"] = "a.png" }, new JsonObject { ["type"] = "text", ["text"] = "Read it." }) },
                    new JsonObject { ["role"] = "assistant", ["content"] = "Plain answer line one.\nLine two." }),
            }) + "\n", new UTF8Encoding(false));
            Check(SyntheticRecordFormats.Detect(chat).Name == "messages", "a messages record");
            var written = SyntheticPages.Write(chat, Path.Combine(folder, "from-chat.jsonl"), options with { AnswerKey = null });
            var transcript = TuningDataFormats.Read(written.Data).Single();
            var parts = transcript.Messages[0].Parts;
            Check(parts.OfType<ChatImage>().Count() == 1 && parts.OfType<ChatText>().Any(t => t.Text.Contains("Read it.", StringComparison.Ordinal)),
                "the old image part replaced by the page, the question kept");
            Check(transcript.Messages[^1].Parts.OfType<ChatText>().Single().Text == "Plain answer line one.\nLine two.", "a plain answer is the drawn text");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void SyntheticParts(Device device)
    {
        string folder = SyntheticFolder("parts");
        try
        {
            File.WriteAllText(Path.Combine(folder, "names.txt"), "سعد\n\nخالد\n", new UTF8Encoding(false));
            var parts = SyntheticPageParts.FromJson(JsonNode.Parse("""
                {
                  "headerEnd": [["N {digits:4}", "R {number:10-12}", "P {pick:x|y}", "D {date:yyyy-MM-dd}", "H {hijri:d/M/yyyy}", "F {field:doc}", "G {field:missing}", "L {line:names.txt}", "{{brace}}"]],
                  "footers": [["never"]],
                  "chances": {"footer": 0},
                  "digits": "٠١٢٣٤٥٦٧٨٩",
                  "nativeDigits": 0
                }
                """)!.AsObject(), folder);
            var options = SyntheticOptions(folder, variants: 1) with { Parts = parts, Scans = ["none"], DateFrom = new DateTime(2025, 3, 1), DateTo = new DateTime(2025, 3, 1) };
            string answer = SyntheticPages.Json(new JsonObject { ["full_text"] = "Body text.", ["doc"] = "D-9" });
            var page = SyntheticPages.Make(answer, options);
            var header = page.Text.Split('\n');
            Check(header[0].Length == 6 && header[0][2..].All(char.IsAsciiDigit), $"digits: {header[0]}");
            Check(header[1] is "R 10" or "R 11" or "R 12" && header[2] is "P x" or "P y", $"number and pick: {header[1]}, {header[2]}");
            Check(header[3] == "D 2025-03-01" && header[4] == "H 1/9/1446", $"the page's date, Gregorian and Umm al-Qura: {header[3]}, {header[4]}");
            Check(header[5] == "F D-9" && !page.Text.Contains("G ", StringComparison.Ordinal), "a field filled, a line with a missing field left out");
            Check(header[6] is "L سعد" or "L خالد" && header[7] == "{brace}" && header[8] == "Body text.", $"a line of a file, braces: {header[6]}, {header[7]}");
            Check(!page.Text.Contains("never", StringComparison.Ordinal), "a part at chance 0 is never drawn");
            Check(SyntheticPages.Make(answer, options).Text == page.Text && SyntheticPages.Make(answer, options, variant: 1).Text != page.Text, "the same page for the same variant, another for another");

            var native = SyntheticPages.Make(answer, options with { Parts = parts with { NativeDigits = 1 } });
            Check(native.Text.Split('\n')[3] == "D ٢٠٢٥-٠٣-٠١" && native.Text.Contains("Body text.", StringComparison.Ordinal), $"the language's digits in the parts, not in the body: {native.Text.Split('\n')[3]}");

            foreach (var (json, why) in new[]
            {
                ("""{"header": [["x"]]}""", "an unknown key"),
                ("""{"footers": [["{nope:1}"]]}""", "an unknown slot"),
                ("""{"footers": [["{digits"]]}""", "a slot not closed"),
                ("""{"chances": {"footer": 2}}""", "a chance above 1"),
                ("""{"chances": {"body": 1}}""", "a chance of no part"),
                ("""{"digits": "0123"}""", "digits not ten"),
                ("""{"nativeDigits": 0.5}""", "native digits with no digits"),
            })
            {
                Throws<InvalidDataException>(() => SyntheticPageParts.FromJson(JsonNode.Parse(json)!.AsObject()), why);
            }

            Throws<ArgumentException>(() => SyntheticPages.Make(answer, options with { Parts = SyntheticPageParts.FromJson(JsonNode.Parse("""{"footers": [["{digits:99}"]]}""")!.AsObject()) }), "digits:99");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void SyntheticSkips(Device device)
    {
        string folder = SyntheticFolder("skips");
        try
        {
            string Record(string text) => SyntheticPages.Json(new JsonObject
            {
                ["conversations"] = new JsonArray(new JsonObject { ["from"] = "human", ["value"] = "<image>" }, new JsonObject { ["from"] = "gpt", ["value"] = text }),
                ["images"] = new JsonArray("x.png"),
            });

            string input = Path.Combine(folder, "mixed.jsonl");
            File.WriteAllLines(input, [Record("نص عربي سليم"), Record("漢字のページ"), Record("   "), Record("Latin text")], new UTF8Encoding(false));
            var options = SyntheticOptions(folder, variants: 2) with { Parts = SyntheticPageParts.Empty };
            var result = SyntheticPages.Write(input, Path.Combine(folder, "out.jsonl"), options);
            Check(result.Records == 4 && result.Pages == 4 && result.Skips.Count == 2, $"{result.Pages} pages, left out: {string.Join(" | ", result.Skips)}");
            Check(result.Skips.Any(s => s.Contains("U+6F22", StringComparison.Ordinal) && s.Contains("line 2", StringComparison.Ordinal)) && result.Skips.Any(s => s.Contains("empty", StringComparison.Ordinal)),
                "each said why and where");

            var arabicOnly = SyntheticPages.Make("نص عربي", options);
            Check(arabicOnly.Description.Contains("DejaVu", StringComparison.Ordinal), $"an Arabic page drawn in the font with Arabic ({arabicOnly.Description})");

            string none = Path.Combine(folder, "none.jsonl");
            File.WriteAllLines(none, [Record("漢字")], new UTF8Encoding(false));
            Throws<InvalidDataException>(() => SyntheticPages.Write(none, Path.Combine(folder, "none-out.jsonl"), options), "every record left out");
            Throws<ArgumentException>(() => SyntheticPages.Make("x", options with { Style = options.Style with { Fonts = [] } }), "no font");
            Throws<NotSupportedException>(() => SyntheticPages.Make("x", options with { Layouts = ["nope"] }), "an unknown layout");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void SyntheticRegistries(Device device)
    {
        Check(new[] { "sharegpt", "messages", "text" }.All(SyntheticRecordFormats.Names.Contains), $"formats {string.Join(", ", SyntheticRecordFormats.Names)}");
        Check(new[] { "digits", "number", "pick", "field", "line", "date", "hijri" }.All(PagePartFillers.Names.Contains), $"slots {string.Join(", ", PagePartFillers.Names)}");
        Throws<NotSupportedException>(() => SyntheticRecordFormats.Get("nope"), "an unknown format");

        PagePartFillers.Register(new FixedFiller());
        try
        {
            var parts = SyntheticPageParts.FromJson(JsonNode.Parse("""{"footers": [["{pick:a} {digits:2}"]]}""")!.AsObject());
            var page = SyntheticPages.Make("Body.", SyntheticOptions(".", variants: 1) with { Parts = parts, Scans = ["none"] });
            string footer = page.Text.Split('\n')[^1];
            Check(footer.StartsWith("PICKED ", StringComparison.Ordinal) && footer.Length == "PICKED ".Length + 2, $"the app's pick, the library's digits: {footer}");
        }
        finally
        {
            PagePartFillers.Unregister("pick");
        }

        Check(PagePartFillers.Get("pick").Summary.StartsWith("{pick:", StringComparison.Ordinal), "unregistered: the library's pick again");
    }

    private sealed class FixedFiller : IPagePartFiller
    {
        public string Name => "pick";

        public string Summary => "always PICKED";

        public string? Fill(string? argument, PagePartContext context) => "PICKED";
    }
}
