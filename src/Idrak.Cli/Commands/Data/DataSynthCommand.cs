// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Nlp;

namespace Idrak.Cli.Commands.Data;

/// <summary>
/// <c>idrak data synth INPUT -o OUT.jsonl --fonts DIR</c>: tuning pages drawn from text (plan 18,
/// <see cref="SyntheticPages"/>): each record's answer text set as pages in the fonts given, around it the parts of a
/// parts file, made to look scanned, written back as records in the input's shape.
/// </summary>
internal sealed class DataSynthCommand : Command
{
    public override string Name => "data synth";

    public override IReadOnlyCollection<string> Aliases => ["synth"];

    public override string Summary => "Tuning pages drawn from text: each record's answer set as scanned-looking pages";

    public override string Usage => """
        INPUT -o OUT.jsonl --fonts DIR [--parts FILE] [-n VARIANTS] [options]

        INPUT is ShareGPT or chat messages records with JSON or text answers (.jsonl or .json), or plain text (a .txt of
        documents between lines of ---, or a folder of .txt files). Each record's answer text is drawn as pages; the
        record is written again with the page as its image and the drawn text, in reading order, as its answer's text.

        Options:
          -o, --out FILE          the records written (.jsonl); the images go to the folder of its name beside it (or
                                  its .zip with --zip), how each page was made to NAME.manifest.jsonl
              --fonts DIR         the fonts (.ttf, .otf, .ttc); each page is set in those that have its characters
              --parts FILE        header, subject, signature and footer parts, with slots such as {digits:6} or {hijri}
                                  (JSON: "headerStart", "headerCenter", "headerEnd", "titles", "subjects", "signatures",
                                  "footers", "chances", "digits", "nativeDigits", "language"); none: the text alone
          -n, --variants N        pages per record (3)
              --layout NAME       a page layout, again for more (letter and plain; a name twice is drawn twice as often)
              --scan PROFILE      a look, again for more, one drawn per page: a name (none: as drawn, clean, office, old,
                                  pale: faint ink, worn: broken strokes, fax, handled: wrinkles, folds, smudges, marked:
                                  pen strikes, crosses, highlighter, damaged: tears, holes, scratches, ripped and
                                  taped, scanner: shadows, streaks, curl, aged: yellowed, foxed, sun-faded, copied:
                                  photocopy, carbon, low toner, photo: a phone photo on a desk, wrecked: heavy damage,
                                  mixed: every effect at a small chance), names
                                  joined with + for several on one page ("damaged+marked"), or one written as text,
                                  "rotate=-1.5..1.5@0.2,fade=0.3..0.7@0.5,taped=1..2@0.5" (all of them, mixed three
                                  times; a name twice, twice as often)
              --grey P            the share of pages in grey (0.3)
              --page SETTINGS     page settings, KEY=VALUE,...: width, height, size, spacing, margin, ink, justify,
                                  color, rule, emblem, stamp, blue-signature, header-font (ranges as MIN..MAX)
              --records NAME      the input's record format: sharegpt, messages or text (found from the input)
              --full-text KEY     the answer's key holding the page text (the first full_text..., else the longest text)
              --prompt TEXT       the question of records made from plain text ("Extract the text.")
              --answer-key KEY    write plain text answers as JSON under KEY
              --image-format F    jpeg (the default) or another registered encoder (png)
              --quality Q         the JPEG quality (90)
              --zip               the images in a zip instead of a folder
              --workers N         pages made at once (one per processor)
              --limit N           read at most N records (a trial)
              --seed N            the data set's seed (0): the same seed and inputs give the same bytes

        Examples:
          idrak data synth train.jsonl -o synthetic/pages.jsonl --fonts fonts --parts parts.json -n 3
          idrak synth letters.txt -o pages.jsonl --fonts fonts --answer-key full_text --scan old --scan office
          idrak data synth train.jsonl -o trial.jsonl --fonts fonts --limit 5 --page "size=20..28,stamp=0.5" --zip
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } =
        ["--out", "--fonts", "--parts", "--variants", "--layout", "--scan", "--grey", "--page", "--records", "--full-text", "--prompt", "--answer-key",
         "--image-format", "--quality", "--workers", "--limit"];

    public override IReadOnlyCollection<string> Flags { get; } = ["--zip"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-o"] = "--out", ["-n"] = "--variants" };

    public override int Run(CommandContext context)
    {
        string input = context.Argument(0, "INPUT");
        string output = context.Option("--out") ?? throw new UsageException("Give -o, --out FILE for the records to write (.jsonl).");
        string fontsFolder = context.Option("--fonts") ?? throw new UsageException("Give --fonts DIR, a folder of .ttf, .otf or .ttc fonts.");
        if (!File.Exists(input))
        {
            throw new UsageException($"The records file {input} is not there (idrak data text writes one).");
        }

        if (!Directory.Exists(fontsFolder))
        {
            throw new UsageException($"The fonts folder {fontsFolder} is not there.");
        }

        var fonts = SyntheticPages.Fonts(fontsFolder);
        if (fonts.Count == 0)
        {
            throw new UsageException($"No font was read from {fontsFolder} (.ttf, .otf, .ttc).");
        }

        SyntheticPageOptions options;
        try
        {
            var defaults = new SyntheticPageOptions { Style = new PageStyle { Fonts = fonts } };
            var layouts = context.Options("--layout").SelectMany(l => l.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)).ToList();

            // A profile written as text holds commas; names may be given as a list.
            var scans = context.Options("--scan").SelectMany(s => s.Contains('=') ? [s] : s.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)).ToList();
            options = defaults with
            {
                Style = SyntheticPages.Style(defaults.Style, context.Option("--page")),
                Parts = context.Option("--parts") is { } parts ? SyntheticPageParts.Read(parts) : SyntheticPageParts.Empty,
                Variants = context.IntOption("--variants", defaults.Variants),
                Layouts = layouts.Count > 0 ? layouts : defaults.Layouts,
                Scans = scans.Count > 0 ? scans : defaults.Scans,
                Grey = context.Option("--grey") is { } grey ? double.Parse(grey, NumberStyles.Float, CultureInfo.InvariantCulture) : defaults.Grey,
                Seed = context.Seed ?? 0,
                Format = context.Option("--records"),
                FullTextKey = context.Option("--full-text"),
                Prompt = context.Option("--prompt") ?? defaults.Prompt,
                AnswerKey = context.Option("--answer-key"),
                ImageFormat = context.Option("--image-format") ?? defaults.ImageFormat,
                Quality = context.IntOption("--quality", defaults.Quality),
                Zip = context.Flag("--zip"),
                Workers = context.IntOption("--workers", 0),
                Limit = context.IntOption("--limit", 0),
            };
        }
        catch (Exception e) when (e is ArgumentException or FormatException or InvalidDataException or NotSupportedException or FileNotFoundException)
        {
            throw new UsageException(e.Message);
        }

        SyntheticPagesResult result;
        using (var line = new ProgressLine(context, $"drawing pages from {Path.GetFileName(input)}", unit: ProgressUnit.Items))
        {
            try
            {
                result = SyntheticPages.Write(input, output, options, new ReportPages(line));
            }
            catch (Exception e) when (e is ArgumentException or InvalidDataException or NotSupportedException or FileNotFoundException)
            {
                throw new UsageException(e.Message);
            }

            line.Finish();
        }

        context.Write(string.Create(CultureInfo.InvariantCulture,
            $"{result.Pages:N0} pages from {result.Records:N0} records in {result.Elapsed.TotalSeconds:F1} s: {result.Data}, images in {result.Images}, manifest {result.Manifest}"));
        // The mix the pages got, from the manifest: a line per look, its share of the pages.
        var looks = File.ReadLines(result.Manifest).Select(l => (string?)JsonNode.Parse(l)?["profile"] ?? "?")
            .GroupBy(p => p).OrderByDescending(g => g.Count()).ThenBy(g => g.Key, StringComparer.Ordinal).ToList();
        if (looks.Count > 0)
        {
            context.Write("looks: " + string.Join(", ", looks.Select(g => string.Create(CultureInfo.InvariantCulture,
                $"{(g.Key.Length > 24 ? g.Key[..24] + "…" : g.Key)} {100.0 * g.Count() / Math.Max(1, result.Pages):F0}%"))));
        }

        foreach (string skip in result.Skips.Take(5))
        {
            context.Write($"left out: {skip}");
        }

        if (result.Skips.Count > 5)
        {
            context.Write(string.Create(CultureInfo.InvariantCulture, $"and {result.Skips.Count - 5:N0} more left out (the first 50 are in the result)"));
        }

        context.WriteJson(new JsonObject
        {
            ["input"] = input,
            ["records"] = result.Records,
            ["pages"] = result.Pages,
            ["data"] = result.Data,
            ["images"] = result.Images,
            ["manifest"] = result.Manifest,
            ["seconds"] = Math.Round(result.Elapsed.TotalSeconds, 2),
            ["skipped"] = new JsonArray([.. result.Skips.Select(s => (JsonNode)s)]),
        });
        return ExitCodes.Ok;
    }

    // The library's progress onto the progress line.
    private sealed class ReportPages(ProgressLine line) : IProgress<SyntheticPagesProgress>
    {
        public void Report(SyntheticPagesProgress value)
        {
            line.Total = value.Total;
            line.Report(value.Pages);
        }
    }
}
