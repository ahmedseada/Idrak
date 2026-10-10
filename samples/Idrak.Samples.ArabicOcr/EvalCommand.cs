// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Nlp;

namespace Idrak.Samples.ArabicOcr;

/// <summary>
/// <c>eval</c>: character and word error rates (<see cref="TuningMetrics"/> "cer" and "wer") of either reader, per page and
/// in total (the errors of every page over the length of every reference, as the metrics add up). Both texts are compared
/// as the app keeps transcriptions (<see cref="OcrText.Normalize"/>: NFC, white space and line breaks as single spaces).
/// </summary>
internal static class EvalCommand
{
    private const string Help = """
        eval <image|folder|data.json>... --reader lines|vlm --model DIR [--truth DIR|FILE] [-j]
          The expected text of each image: --truth DIR (NAME.txt there), --truth FILE (one image's text, or a data file
          whose pages are found by their image), else NAME.txt beside the image (cut's lines: add --single-line for the
          line reader). A data file (.json, .jsonl: ShareGPT or chat messages) is read page by page against its answers,
          its prompt given to the vision-language reader (unless --prompt); --images DIR|ZIP where its images are.
          --truth-text raw|values   compare the texts as they are (raw), or a JSON answer's text values in order (values;
                                    applied to the reading too when it is JSON)
          -o FILE                   the report (JSON with -j, else the table) to FILE as well
          --limit N                 the first N pages only
          The readers' options are read's (see read --help).
        """;

    public static int Run(string[] args, TextWriter output, TextWriter error)
    {
        if (OcrApp.Context(args, output, error,
                ["--reader", "--model", "--truth", "--truth-text", "--images", "--limit", "--out", .. Readers.LineValues, .. Readers.VlmValues],
                [.. Readers.LineFlags, .. Readers.VlmFlags], Help) is not { } context)
        {
            return 0;
        }

        var a = context.Args;
        string mode = a.Option("--truth-text") ?? "raw";
        _ = OcrText.AnswerText("", mode);
        string? truth = a.Option("--truth"), images = a.Option("--images");
        Dictionary<string, Page>? truthPages = truth is not null && File.Exists(truth) && Pages.IsDataFile(truth) ? Pages.ByImage(truth, images) : null;
        using var reader = Readers.Create(a.Required("--reader", "lines or vlm"), context);
        var cer = TuningMetrics.Get(TuningMetrics.CharacterErrorRate);
        var wer = TuningMetrics.Get(TuningMetrics.WordErrorRate);
        var alphabet = (reader as LinesReader)?.Alphabet.ToHashSet(StringComparer.Ordinal);
        var unknown = new SortedDictionary<string, int>(StringComparer.Ordinal);
        var rows = new JsonArray();
        var table = new List<string> { $"{"page",-32} {"CER",8} {"WER",8} {"chars",7} {"words",6}" };
        TuningScore totalCer = default, totalWer = default;
        int skipped = 0;
        foreach (var page in Pages.Read(a.Words, images).Take(a.Integer("--limit", int.MaxValue, 1)))
        {
            string? expected = page.Truth ?? Expected(page, truth, truthPages);
            if (expected is null)
            {
                skipped++;
                context.Say($"{page.Source}: no expected text, skipped");
                continue;
            }

            var reading = reader.Read(page);
            string answer = OcrText.Normalize(OcrText.AnswerText(reading.Text, mode));
            string reference = OcrText.Normalize(OcrText.AnswerText(expected, mode));
            var c = cer.Score(answer, reference);
            var w = wer.Score(answer, reference);
            totalCer += c;
            totalWer += w;
            if (alphabet is not null)
            {
                foreach (var rune in reference.EnumerateRunes().Where(r => !alphabet.Contains(r.ToString())))
                {
                    unknown[rune.ToString()] = unknown.GetValueOrDefault(rune.ToString()) + 1;
                }
            }

            table.Add(FormattableString.Invariant($"{Shorten(page.Name, 32),-32} {c.Value,8:F4} {w.Value,8:F4} {c.Denominator,7} {w.Denominator,6}"));
            rows.Add(new JsonObject
            {
                ["name"] = page.Name, ["source"] = page.Source, ["cer"] = c.Value, ["wer"] = w.Value,
                ["character_errors"] = c.Numerator, ["characters"] = c.Denominator, ["word_errors"] = w.Numerator, ["words"] = w.Denominator,
                ["text"] = answer, ["truth"] = reference, ["seconds"] = reading.Details["seconds"]?.DeepClone(),
            });
        }

        table.Add(FormattableString.Invariant($"{"total",-32} {totalCer.Value,8:F4} {totalWer.Value,8:F4} {totalCer.Denominator,7} {totalWer.Denominator,6}"));
        var json = new JsonObject
        {
            ["reader"] = reader.Name,
            ["truth_text"] = mode,
            ["pages"] = rows,
            ["skipped"] = skipped,
            ["total"] = new JsonObject
            {
                ["cer"] = totalCer.Value, ["wer"] = totalWer.Value, ["character_errors"] = totalCer.Numerator, ["characters"] = totalCer.Denominator,
                ["word_errors"] = totalWer.Numerator, ["words"] = totalWer.Denominator,
            },
            ["unknown_characters"] = new JsonObject([.. unknown.Select(u => KeyValuePair.Create(u.Key, (JsonNode?)u.Value))]),
        };
        if (unknown.Count > 0)
        {
            context.Say("characters the recognizer has never seen (it cannot read them; they count as errors): "
                        + string.Join(" ", unknown.Select(u => string.Create(CultureInfo.InvariantCulture, $"'{u.Key}'×{u.Value}"))));
        }

        string report = context.Json ? json.ToJsonString(Json.Indented) : string.Join("\n", table);
        context.Output.WriteLine(OcrText.EndLines(report));
        if (a.Option("--out") is { } file)
        {
            OcrText.Write(file, report + "\n");
        }

        return rows.Count > 0 ? 0 : 1;
    }

    // The expected text of an image: --truth DIR/NAME.txt, --truth FILE (one image), a data file's page with the same
    // image, else NAME.txt beside the image.
    private static string? Expected(Page page, string? truth, Dictionary<string, Page>? truthPages)
    {
        if (truthPages is not null)
        {
            return truthPages.TryGetValue(page.Image.Hash, out var known) ? known.Truth : null;
        }

        string? file = truth is null ? LineFiles.Of(page.Source, LineFiles.Corrected)
            : Directory.Exists(truth) ? Path.Combine(truth, page.Name + ".txt")
            : truth;
        return file is not null && File.Exists(file) && OcrText.ReadTranscription(file) is { Length: > 0 } text ? text : null;
    }

    private static string Shorten(string text, int width) => text.Length <= width ? text : "…" + text[^(width - 1)..];
}
