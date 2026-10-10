// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The OCR app's tests, on generated data only (Synthetic: letters drawn as strokes, no fonts) and the repository's tiny
// Gemma 3 fixture (random weights: the vision-language reader's plumbing, not its text). The app's commands run in
// process. A plain runner like the repository's: IDRAK_FILTER runs the tests whose name contains it, IDRAK_DEVICES the
// device (its first entry; the CPU by default).
//
//   IDRAK_DEVICES=cpu dotnet run -c Release --project samples/Idrak.Samples.ArabicOcr.Tests
//   IDRAK_DEVICES=cpu IDRAK_FILTER="ocr train" dotnet run -c Release --project samples/Idrak.Samples.ArabicOcr.Tests

using System.Diagnostics;
using System.Globalization;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Nlp.Abstractions;
using Idrak.Samples.ArabicOcr;
using Idrak.Samples.ArabicOcr.Tests;

string device = Environment.GetEnvironmentVariable("IDRAK_DEVICES") is { Length: > 0 } devices ? devices.Split(',')[0].Trim() : "cpu";
string root = Path.Combine(Path.GetTempPath(), $"idrak-ocr-tests-{Environment.ProcessId}");
Directory.CreateDirectory(root);
string tinyGemma = Path.Combine(RepositoryRoot(), "tests", "Idrak.Tests", "data", "vlm", "tiny-gemma3");
const string Alef = "ا", Beh = "ب", Teh = "ت", Theh = "ث", Dal = "د", Reh = "ر", Seen = "س", Meem = "م";
string? trained = null;
var timings = new List<string>();

(string Name, Action Run)[] tests =
[
    ("ocr segment: four lines found top to bottom, dots joined to their line", () =>
    {
        string[] text = [Theh + Teh + Dal + " " + Beh + Seen, Alef + Reh + Meem, Teh + Theh + " " + Theh + Teh, Seen + Beh + Dal + Reh];
        var (page, rows) = Synthetic.Page(text, new Random(1));
        var (lines, skew) = LineSegmenter.Find(page);
        Check(skew == 0, $"skew {skew}");
        Check(lines.Count == 4, $"{lines.Count} lines");
        for (int i = 0; i < 4; i++)
        {
            // Each box inside its line's cell and the gaps around it (padding included), its middle in the cell.
            int middle = lines[i].Y + lines[i].Height / 2;
            Check(lines[i].Index == i + 1 && lines[i].Y >= rows[i].Top - 8 && lines[i].Y + lines[i].Height <= rows[i].Bottom + 8 && middle > rows[i].Top && middle < rows[i].Bottom,
                $"line {i + 1}: rows {lines[i].Y}..{lines[i].Y + lines[i].Height}, drawn {rows[i].Top}..{rows[i].Bottom}");
        }
    }),
    ("ocr segment: a page turned by two degrees is deskewed and still gives its lines", () =>
    {
        string[] text = [Theh + Teh + Dal + Beh + Seen + Meem + Reh + Alef + Teh + Seen, Alef + Reh + Meem + Seen + Seen + Beh + Dal + Teh,
            Teh + Theh + Theh + Teh + Dal + Dal + Seen + Meem, Seen + Beh + Dal + Reh + Meem + Alef + Teh + Theh];
        var (page, _) = Synthetic.Page(text, new Random(2), skewDegrees: 2);
        var (lines, skew) = LineSegmenter.Find(page);
        Check(Math.Abs(skew - 2) <= 0.41, $"skew measured {skew}");
        Check(lines.Count == 4, $"{lines.Count} lines");
    }),
    ("ocr text: reading order reverses digits and Latin runs only, and undoes itself", () =>
    {
        string logical = Beh + Alef + " 123 " + Seen + " abc de " + Meem + " ١٢";
        string columns = ReadingOrder.ToColumns(logical);
        Check(columns == Beh + Alef + " 321 " + Seen + " ed cba " + Meem + " ٢١", $"columns: {columns}");
        Check(ReadingOrder.FromColumns(columns) == logical, "not its own inverse");
        Check(OcrText.Normalize("﻿  a\r\n\tb  ") == "a b" && OcrText.Normalize("é") == "é", "normalization");
    }),
    ("ocr text: JSON answers give their text values in order; drafts snap to the page's text in order", () =>
    {
        string answer = "```json\n{\"title\": \"" + Beh + Alef + "\", \"items\": [{\"text\": \"" + Seen + Meem + "\"}, 12], \"empty\": \"\"}\n```";
        Check(OcrText.AnswerText(answer, "values") == Beh + Alef + "\n" + Seen + Meem + "\n12", $"values: {OcrText.AnswerText(answer, "values")}");
        Check(OcrText.AnswerText("not json", "values") == "not json" && OcrText.AnswerText(answer, "raw") == answer, "raw and plain text");
        string page = "the first line here then a second line and the third one";
        var aligned = TextAlignment.Align(["the frst line hre", "zzzzzzzzzzzzzzzzzzzz", "a secnd line", "the thrd one"], page);
        Check(aligned[0]?.Text == "the first line here" && aligned[1] is null && aligned[2]?.Text == "a second line" && aligned[3]?.Text == "the third one",
            $"aligned: {string.Join(" | ", aligned.Select(a => a?.Text ?? "(none)"))}");
        Check(aligned[0]!.Value.Match is > 0.8 and < 1, $"match {aligned[0]!.Value.Match}");
    }),
    ("ocr cut: lines and empty .txt files; a filled .txt is never overwritten", () =>
    {
        string folder = Fresh("cut"), lines = Path.Combine(folder, "lines");
        var (page, _) = Synthetic.Page([Theh + Teh + Dal, Alef + Reh + Meem + Seen, Beh + Seen + Dal], new Random(3));
        Synthetic.Save(Path.Combine(folder, "page-001.png"), page);
        var (code, _, error) = App("cut", Path.Combine(folder, "page-001.png"), "-o", lines);
        Check(code == 0, error);
        var pngs = Directory.GetFiles(lines, "*.png").Select(Path.GetFileName).Order(StringComparer.Ordinal).ToArray();
        Check(pngs.SequenceEqual(["page-001-line-01.png", "page-001-line-02.png", "page-001-line-03.png"]), string.Join(", ", pngs));
        Check(pngs.All(p => File.Exists(Path.Combine(lines, Path.ChangeExtension(p!, ".txt"))) && new FileInfo(Path.Combine(lines, Path.ChangeExtension(p!, ".txt"))).Length == 0), "empty .txt beside each line");
        string filled = Path.Combine(lines, "page-001-line-02.txt");
        File.WriteAllText(filled, Alef + Reh + Meem + Seen + "\n");
        byte[] before = File.ReadAllBytes(Path.Combine(lines, "page-001-line-01.png"));
        (code, _, error) = App("cut", folder, "-o", lines);
        Check(code == 0, error);
        Check(File.ReadAllText(filled) == Alef + Reh + Meem + Seen + "\n", "the filled .txt changed");
        Check(File.ReadAllBytes(Path.Combine(lines, "page-001-line-01.png")).SequenceEqual(before), "the line image changed on a second run");
        Check(Directory.GetFiles(lines).Length == 6, $"{Directory.GetFiles(lines).Length} files after the second run");
    }),
    ("ocr train: CER falls substantially within a few epochs on a tiny synthetic set", () =>
    {
        string model = TrainedModel();
        var json = (JsonObject)JsonNode.Parse(File.ReadAllText(Path.Combine(model, Recognizer.SettingsFile)))!;
        var history = json["training"]!["history"]!.AsArray();
        double first = (double)history[0]!["cer"]!, best = (double)json["training"]!["cer"]!;
        Check(best <= 0.2 && best <= first / 3, $"cer {first} at the first epoch, best {best}");
        Check(json["alphabet"]!.AsArray().Count == 9 && (string?)json["direction"] == "rtl", "alphabet of 8 letters and the space, right to left");
    }),
    ("ocr read: lines reader on a synthetic page returns its text in logical order", () =>
    {
        string model = TrainedModel();
        string folder = Fresh("read");
        string[] text = [Seen + Beh + Dal + " " + Reh + Meem, Theh + Alef + Teh + Dal, Meem + Seen + " " + Alef + Beh];
        var (page, _) = Synthetic.Page(text, new Random(4));
        string png = Path.Combine(folder, "page.png");
        Synthetic.Save(png, page);
        var (code, output, error) = App("read", png, "--reader", "lines", "--model", model, "-j");
        Check(code == 0, error);
        var pages = JsonNode.Parse(output)!["pages"]!.AsArray();
        var lines = pages[0]!["lines"]!.AsArray();
        Check(lines.Count == 3, $"{lines.Count} lines: {output}");
        string read = (string)pages[0]!["text"]!;
        double cer = TuningMetrics.Get("cer").Score(read, string.Join("\n", text)).Value;
        Check(cer <= 0.1, $"read '{read}' for '{string.Join("|", text)}' (cer {cer})");
        // Logical order: the reading is far from each line's letters reversed (what an unflipped reading would give).
        string reversed = string.Join("\n", text.Select(t => new string([.. t.Reverse()])));
        Check(TuningMetrics.Get("cer").Score(read, reversed).Value > 3 * cer + 0.2, "the reading is in visual (reversed) order");
        Check(lines.All(l => (double)l!["score"]! is > 0 and <= 1 && (int)l["box"]!["height"]! > 0), "boxes and scores");
        // The text output: the same lines, and -o writes them as UTF-8 without a BOM, lines ending as the console's.
        string outFile = Path.Combine(folder, "page.txt");
        (code, output, error) = App("read", png, "--reader", "lines", "--model", model, "--decoder", "beam", "-o", outFile);
        Check(code == 0, error);
        byte[] bytes = File.ReadAllBytes(outFile);
        Check(bytes.Length > 3 && !(bytes[0] == 0xEF && bytes[1] == 0xBB), "a BOM");
        Check(File.ReadAllText(outFile) == OcrText.EndLines(read + "\n") || TuningMetrics.Get("cer").Score(File.ReadAllText(outFile).Trim(), read).Value <= 0.1, "beam's file");
    }),
    ("ocr eval: per line and total CER/WER are TuningMetrics' over the same texts; unknown characters reported", () =>
    {
        string model = TrainedModel();
        string lines = Path.Combine(root, "data", "eval");
        var clock = Stopwatch.StartNew();
        var (code, output, error) = App("eval", lines, "--reader", "lines", "--model", model, "--single-line", "-j");
        timings.Add(string.Create(CultureInfo.InvariantCulture, $"eval read {Directory.GetFiles(lines, "*.png").Length} line images in {clock.Elapsed.TotalSeconds:F2} s (loading the model included)"));
        Check(code == 0, error);
        var json = JsonNode.Parse(output)!;
        var pages = json["pages"]!.AsArray();
        Check(pages.Count == Directory.GetFiles(lines, "*.png").Length, $"{pages.Count} lines scored");
        var pairs = pages.Select(p => ((string)p!["text"]!, (string)p["truth"]!)).ToArray();
        foreach (var p in pages)
        {
            Check((string)p!["truth"]! == OcrText.ReadTranscription(Path.Combine(lines, (string)p["name"]! + ".txt")), "truth is the .txt");
        }

        var cer = TuningMetrics.Score("cer", pairs);
        var wer = TuningMetrics.Score("wer", pairs);
        Check((double)json["total"]!["cer"]! == cer.Value && (double)json["total"]!["wer"]! == wer.Value
              && (double)json["total"]!["characters"]! == cer.Denominator, $"total {json["total"]!.ToJsonString()} against {cer} {wer}");
        // A truth with a letter the recognizer never saw: reported, counted as an error, no crash.
        string odd = Fresh("odd");
        string png = Path.Combine(odd, "x.png");
        Synthetic.Save(png, Synthetic.LineImage(Beh + Alef + Seen, new Random(5)));
        File.WriteAllText(Path.Combine(odd, "x.txt"), Beh + Alef + Seen + "ك");
        (code, output, error) = App("eval", png, "--reader", "lines", "--model", model, "--single-line", "-j");
        Check(code == 0 && JsonNode.Parse(output)!["unknown_characters"]!["ك"] is not null, $"{code} {output} {error}");
    }),
    ("ocr train: drafts are left out unless --include-drafts; accept renames aligned drafts only", () =>
    {
        string folder = Fresh("drafts");
        var random = new Random(6);
        for (int i = 0; i < 5; i++)
        {
            string text = Synthetic.Text(random, 2, 4);
            Synthetic.Save(Path.Combine(folder, $"l{i}.png"), Synthetic.LineImage(text, random));
            string kind = i < 2 ? ".txt" : i < 4 ? ".aligned.txt" : ".draft.txt";
            File.WriteAllText(Path.Combine(folder, $"l{i}{kind}"), text);
        }

        File.WriteAllText(Path.Combine(folder, "l2.txt"), "");                                // empty: not transcribed
        string[] small = ["--height", "16", "--channels", "4,8,8", "--hidden", "8", "--layers", "1", "--epochs", "1", "--augment", "none", "-d", device, "-j"];
        var (code, output, error) = App(["train", "--data", folder, "-o", Path.Combine(folder, "m1"), .. small]);
        Check(code == 0 && (int)JsonNode.Parse(output)!["train_lines"]! == 2, $"corrected only: {output} {error}");
        (code, output, error) = App(["train", "--data", folder, "-o", Path.Combine(folder, "m2"), "--include-drafts", .. small]);
        Check(code == 0 && (int)JsonNode.Parse(output)!["train_lines"]! == 5, $"with drafts: {output} {error}");
        File.WriteAllText(Path.Combine(folder, "l0.aligned.txt"), "something else");        // beside a corrected .txt
        (code, output, error) = App("accept", folder, "-j");
        Check(code == 0 && (int)JsonNode.Parse(output)!["accepted"]! == 2 && (int)JsonNode.Parse(output)!["kept"]! == 2, $"accept: {output} {error}");
        Check(File.Exists(Path.Combine(folder, "l4.draft.txt")) && !File.Exists(Path.Combine(folder, "l4.txt")), "a plain draft accepted without --drafts");
        Check(File.ReadAllText(Path.Combine(folder, "l0.txt")) != "something else", "a corrected .txt replaced");
        Check(OcrText.ReadTranscription(Path.Combine(folder, "l2.txt")).Length > 0 && !File.Exists(Path.Combine(folder, "l2.aligned.txt")), "the empty .txt not filled");
    }),
    ("ocr vlm: the vision-language reader reads a page with the tiny Gemma 3 (plumbing; random weights)", () =>
    {
        string folder = Fresh("vlm");
        string png = Path.Combine(folder, "page.png");
        Synthetic.Save(png, Synthetic.Page([Beh + Alef + Seen + Seen + Dal + Teh + Theh + Meem + Seen + Seen, Meem + Reh], new Random(7)).Page);
        var (code, output, error) = App("read", png, "--reader", "vlm", "--model", tinyGemma, "--max-tokens", "6", "--grayscale", "-j");
        Check(code == 0, error);
        var page = JsonNode.Parse(output)!["pages"]![0]!;
        Check((int)page["prompt_tokens"]! > 0 && (int)page["generated_tokens"]! is > 0 and <= 6 && ((string)page["image_transforms"]!).Contains("grayscale"),
            $"read: {output}");
        int plain = (int)page["prompt_tokens"]!;
        (code, output, error) = App("read", png, "--reader", "vlm", "--model", tinyGemma, "--max-tokens", "2", "--pan-and-scan", "--vision-option", "pan_and_scan_min_crop_size=8", "-j");
        Check(code == 0 && (int)JsonNode.Parse(output)!["pages"]![0]!["prompt_tokens"]! > plain, $"pan and scan adds crops: {output} {error}");
        (code, output, _) = App("read", png, "--reader", "vlm", "--model", tinyGemma, "--max-tokens", "3");
        Check(code == 0, "text output");
        (code, _, error) = App("read", png, "--reader", "vlm", "--model", tinyGemma, "--vision-option", "tiles=2");
        Check(code != 0 && error.Contains("tiles"), $"an unknown vision option: {code} {error}");
    }),
    ("ocr data: eval reads a ShareGPT file page by page against its answers (zip of images, JSON values)", () =>
    {
        string folder = Fresh("data");
        var (data, zip, answers) = ShareGpt(folder, 2);
        var (code, output, error) = App("eval", data, "--images", zip, "--reader", "vlm", "--model", tinyGemma, "--max-tokens", "4", "--truth-text", "values", "-j");
        Check(code == 0, error);
        var json = JsonNode.Parse(output)!;
        var pages = json["pages"]!.AsArray();
        Check(pages.Count == 2 && (string)pages[0]!["source"]! == "train.json #1", $"pages: {output}");
        double characters = answers.Sum(a => OcrText.Normalize(OcrText.AnswerText(a, "values")).EnumerateRunes().Count());
        Check((double)json["total"]!["characters"]! == characters, $"characters {json["total"]!["characters"]} against {characters}");
        Check((string)pages[1]!["truth"]! == OcrText.Normalize(OcrText.AnswerText(answers[1], "values")), "the answer's values");

        // The same pages as chat "messages" JSON Lines (as ocr-images-sft.jsonl): detected, read page by page.
        string jsonl = Path.Combine(folder, "sft.jsonl");
        File.WriteAllLines(jsonl, answers.Select((answer, i) => new JsonObject
        {
            ["messages"] = new JsonArray(
                new JsonObject { ["role"] = "user", ["content"] = new JsonArray(new JsonObject { ["type"] = "image", ["path"] = $"images/scan-{i + 1}.png" }, new JsonObject { ["type"] = "text", ["text"] = "Read this document." }) },
                new JsonObject { ["role"] = "assistant", ["content"] = answer }),
        }.ToJsonString()));
        (code, output, error) = App("eval", jsonl, "--images", zip, "--reader", "vlm", "--model", tinyGemma, "--max-tokens", "4", "--truth-text", "values", "-j");
        Check(code == 0 && (double)JsonNode.Parse(output)!["total"]!["characters"]! == characters && (string)JsonNode.Parse(output)!["pages"]![1]!["source"]! == "sft.jsonl #2",
            $"messages: {output} {error}");
    }),
    ("ocr cut: --prefill vlm writes drafts, snapped to a ShareGPT page's text; corrected lines kept", () =>
    {
        string folder = Fresh("prefill"), lines = Path.Combine(folder, "lines");
        var (data, zip, answers) = ShareGpt(folder, 1);
        // First a plain cut, and one line corrected by hand.
        var (code, _, error) = App("cut", data, "--images", zip, "-o", lines);
        Check(code == 0, error);
        string corrected = Path.Combine(lines, "train-0001-line-02.txt");
        File.WriteAllText(corrected, "hand made");
        foreach (string empty in Directory.GetFiles(lines, "*.txt").Where(f => f != corrected))
        {
            File.Delete(empty);
        }

        (code, var output, error) = App("cut", data, "--images", zip, "-o", lines, "--prefill", "vlm", "--model", tinyGemma, "--max-tokens", "4", "--max-error", "1", "-j");
        Check(code == 0, error);
        var json = JsonNode.Parse(output)!;
        Check((int)json["lines"]! == 3 && (int)json["corrected_kept"]! == 1 && (int)json["aligned"]! + (int)json["drafts"]! == 2
              && (int)json["aligned"]! >= 1 && Directory.GetFiles(lines, "*.aligned.txt").Length == (int)json["aligned"]!, $"cut: {output}");
        Check(File.ReadAllText(corrected) == "hand made", "the corrected line changed");
        string truth = OcrText.Normalize(OcrText.AnswerText(answers[0], "values"));
        foreach (string file in Directory.GetFiles(lines, "*.aligned.txt"))
        {
            Check(truth.Contains(OcrText.ReadTranscription(file), StringComparison.Ordinal), $"{Path.GetFileName(file)} is not a span of the page's text");
        }

        Check(!Directory.GetFiles(lines, "*.txt").Any(f => f.EndsWith("-line-01.txt", StringComparison.Ordinal) || f.EndsWith("-line-03.txt", StringComparison.Ordinal)),
            "a draft written as a corrected .txt");
        // Again: the drafts are kept, nothing is read twice.
        (code, output, error) = App("cut", data, "--images", zip, "-o", lines, "--prefill", "vlm", "--model", tinyGemma, "--max-tokens", "4", "-j");
        Check(code == 0 && (int)JsonNode.Parse(output)!["aligned"]! + (int)JsonNode.Parse(output)!["drafts"]! == 0, $"second run: {output} {error}");
    }),
    ("ocr tune-vlm prints the idrak tune commands; bad options exit 2", () =>
    {
        var (code, output, error) = App("tune-vlm", "--model", "m", "--data", "train.json", "--eval", "val.json", "--images", "downloaded_images.zip", "--grayscale");
        Check(code == 0 && output.Contains(" tune -P $plugin -b ") && output.Contains("--metric cer") && output.Contains("tune evaluate") && output.Contains("downloaded_images.zip"), output + error);
        (code, _, error) = App("read", "x.png", "--reader", "lines", "--nope");
        Check(code == 2 && error.Contains("--nope"), error);
        (code, _, error) = App("read", "x.png", "--reader", "other", "--model", "m");
        Check(code == 2 && error.Contains("lines or vlm"), error);
    }),
];

int failed = 0, ran = 0;
foreach (var (name, run) in tests.Where(t => Environment.GetEnvironmentVariable("IDRAK_FILTER") is not { } f || t.Name.Contains(f, StringComparison.OrdinalIgnoreCase)))
{
    ran++;
    var clock = Stopwatch.StartNew();
    try
    {
        run();
        Console.WriteLine($"  PASS {name} ({clock.Elapsed.TotalSeconds:F1} s)");
    }
    catch (Exception ex)
    {
        failed++;
        Console.WriteLine($"  FAIL {name}: {ex.Message}");
    }
}

foreach (string line in timings)
{
    Console.WriteLine($"  {line}");
}

try
{
    Directory.Delete(root, recursive: true);
}
catch (IOException)
{
}

Console.WriteLine($"{ran - failed} passed, {failed} failed (device {device})");
return failed == 0 ? 0 : 1;

// The app's command run in process: exit code, output, error.
(int Code, string Output, string Error) App(params string[] args)
{
    var output = new StringWriter();
    var error = new StringWriter();
    bool hasDevice = args.Contains("-d") || args.Contains("--device") || args.Length == 0 || args[0] == "tune-vlm" || args[0] == "accept";
    int code = OcrApp.Run(hasDevice ? args : [.. args, "--device", device], output, error);
    return (code, output.ToString(), error.ToString());
}

// A tiny synthetic set (seeded), made as the owner makes theirs: pages of four lines cut by the app (cut), each line's
// .txt then written with its known text; 160 training and 32 evaluation lines. A small recognizer is trained on it once.
string TrainedModel()
{
    if (trained is not null)
    {
        return trained;
    }

    var random = new Random(11);
    foreach (var (set, pages) in new[] { ("train", 40), ("eval", 8) })
    {
        string scans = Path.Combine(root, "scans", set), folder = Path.Combine(root, "data", set);
        Directory.CreateDirectory(scans);
        var texts = new Dictionary<string, string>();
        for (int p = 1; p <= pages; p++)
        {
            string[] lines = [.. Enumerable.Range(0, 4).Select(_ => Synthetic.Text(random, 3, 7))];
            Synthetic.Save(Path.Combine(scans, $"{set}-{p:D3}.png"), Synthetic.Page(lines, random).Page);
            for (int l = 0; l < 4; l++)
            {
                texts[$"{set}-{p:D3}-line-{l + 1:D2}"] = lines[l];
            }
        }

        var (cut, _, cutError) = App("cut", scans, "-o", folder);
        Check(cut == 0, cutError);
        Check(Directory.GetFiles(folder, "*.png").Length == texts.Count, $"cut found {Directory.GetFiles(folder, "*.png").Length} lines of {texts.Count}");
        foreach (var (name, text) in texts)
        {
            File.WriteAllText(Path.Combine(folder, name + ".txt"), text + "\n");
        }
    }

    string model = Path.Combine(root, "model");
    var clock = Stopwatch.StartNew();
    var (code, output, error) = App("train", "--data", Path.Combine(root, "data", "train"), "--eval", Path.Combine(root, "data", "eval"), "-o", model,
        "--height", "16", "--channels", "16,32,48", "--hidden", "48", "--layers", "1", "--epochs", "12", "--batch", "8", "--lr", "0.003",
        "--augment", "shift(pixels=1), noise(std=0.02)", "--seed", "3", "-j");
    Check(code == 0, error);
    var summary = JsonNode.Parse(output)!;
    timings.Add(string.Create(CultureInfo.InvariantCulture,
        $"trained in {clock.Elapsed.TotalSeconds:F1} s: {summary["seconds_per_epoch"]} s an epoch over 160 lines (and 32 evaluated), best cer {summary["cer"]} at epoch {summary["best_epoch"]}"));
    Console.Error.Write(error);
    return trained = model;
}

// A ShareGPT file of `pages` synthetic pages (train.json) with its images in a zip, each answer a JSON page.
(string Data, string Zip, string[] Answers) ShareGpt(string folder, int pages)
{
    var random = new Random(8);
    string zip = Path.Combine(folder, "images.zip");
    var records = new JsonArray();
    var answers = new List<string>();
    using (var archive = System.IO.Compression.ZipFile.Open(zip, System.IO.Compression.ZipArchiveMode.Create))
    {
        for (int p = 1; p <= pages; p++)
        {
            string[] text = [Synthetic.Text(random, 3, 5), Synthetic.Text(random, 3, 5), Synthetic.Text(random, 3, 5)];
            string png = Path.Combine(folder, $"scan-{p}.png");
            Synthetic.Save(png, Synthetic.Page(text, random).Page);
            System.IO.Compression.ZipFileExtensions.CreateEntryFromFile(archive, png, $"images/scan-{p}.png");
            File.Delete(png);
            string answer = new JsonObject { ["page"] = p, ["lines"] = new JsonArray([.. text.Select(t => (JsonNode)t)]) }.ToJsonString();
            answers.Add(answer);
            records.Add(new JsonObject
            {
                ["conversations"] = new JsonArray(
                    new JsonObject { ["from"] = "human", ["value"] = "<image>Read this document." },
                    new JsonObject { ["from"] = "gpt", ["value"] = answer }),
                ["images"] = new JsonArray($"images/scan-{p}.png"),
            });
        }
    }

    string data = Path.Combine(folder, "train.json");
    File.WriteAllText(data, records.ToJsonString());
    return (data, zip, [.. answers]);
}

string Fresh(string name)
{
    string folder = Path.Combine(root, name);
    if (Directory.Exists(folder))
    {
        Directory.Delete(folder, recursive: true);
    }

    Directory.CreateDirectory(folder);
    return folder;
}

static string RepositoryRoot()
{
    for (var folder = new DirectoryInfo(AppContext.BaseDirectory); folder is not null; folder = folder.Parent)
    {
        if (File.Exists(Path.Combine(folder.FullName, "Idrak.slnx")))
        {
            return folder.FullName;
        }
    }

    throw new InvalidOperationException("Run from the repository (no Idrak.slnx above the test's folder).");
}

static void Check(bool condition, string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}
