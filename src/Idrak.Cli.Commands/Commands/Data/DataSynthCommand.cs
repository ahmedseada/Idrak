// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Text;

namespace Idrak.Cli.Commands.Data;

/// <summary>
/// <c>idrak data synth INPUT -o OUT.jsonl --fonts DIR</c>: tuning pages drawn from text (plan 18,
/// <see cref="SyntheticPages"/>): each record's answer text set as pages in the fonts given, around it the parts of a
/// parts file, made to look scanned, written back as records in the input's shape; as images, or as PDFs (plan 28's
/// follow-up: <c>--format pdf</c>, a scan-like document, or <c>pdf-vector</c>, a document as software makes it) through
/// any document writer.
/// </summary>
internal sealed class DataSynthCommand : Command
{
    public override string Name => "data synth";

    public override IReadOnlyCollection<string> Aliases => ["synth"];

    public override string Summary => "Tuning pages drawn from text: each record's answer set as scanned-looking pages";

    public override string Usage => $$"""
        INPUT -o OUT.jsonl --fonts DIR [--parts FILE] [-n VARIANTS] [options]

        INPUT is ShareGPT or chat messages records with JSON or text answers (.jsonl or .json), or plain text (a .txt of
        documents between lines of ---, or a folder of .txt files). Each record's answer text is drawn as pages; the
        record is written again with the page as its image and the drawn text, in reading order, as its answer's text.

        Options:
          -o, --out FILE          the records written (.jsonl); the images go to the folder of its name beside it (or
                                  its .zip with --zip), how each page was made to NAME.manifest.jsonl
              --fonts DIR         the fonts ({{FontMapSettings.FontFiles()}}); each
                                  page is set in those that have its characters. A legacy font (letter forms in the
                                  Private Use Area, no shaping tables) is mapped by the library's maps and drawn joined,
                                  its missing digits and marks drawn in and missing letters composed
              --font-map FILE     a font map (JSON), again for more: tried before the library's maps
              --font-maps SET     legacy font settings, KEY=VALUE,...: builtin (on: the library's maps), sidecars (on:
                                  FONT.map.json beside FONT.ttf), digits, marks (on: drawn into mapped fonts lacking them), letters (on: composed),
                                  blank-marks (on: a mark without ink is drawn), slant (0.15), wobble (0.012), seed (7),
                                  digit-pen, digit-height, digit-width, mark-size, mark-pen (font units; measured unless set),
                                  mark-share (0.3), mark-pen-share (0.6), pairs, style2, words (the chances a page
                                  joins letter pairs, writes their second style, writes whole-word ligatures; 0.5 each),
                                  alternates (the chance a page picks a hand's alternates per glyph; 1),
                                  cache (on: a legacy font converted to TrueType once and kept, IDRAK_FONT_CACHE; off;
                                  or a folder)
              --parts FILE        header, subject, signature and footer parts, with slots such as {digits:6} or {hijri}
                                  (JSON: "headerStart", "headerCenter", "headerEnd", "titles", "subjects", "signatures",
                                  "footers", "chances", "digits", "nativeDigits", "language"); none: the text alone
          -n, --count N           pages per record (3)
              --layout NAME       a page layout, again for more (letter, plain, and shapes: shapes drawn by a pen, the
                                  text in the regions they make; a name twice is drawn twice as often)
              --scan PROFILE      a look, again for more, one drawn per page: a name (none: as drawn, clean, office, old,
                                  pale: faint ink, worn: broken strokes, fax, handled: wrinkles, folds, smudges, marked:
                                  pen strikes, crosses, highlighter, damaged: tears, holes, scratches, ripped and
                                  taped, scanner: shadows, streaks, curl, aged: yellowed, foxed, sun-faded, copied:
                                  photocopy, carbon, low toner, photo: a phone photo on a desk, wrecked: heavy damage,
                                  mixed: every effect at a small chance), names
                                  joined with + for several on one page ("damaged+marked"), or one written as text,
                                  "rotate=-1.5..1.5@0.2,fade=0.3..0.7@0.5,taped=1..2@0.5" (all of them, mixed three
                                  times; a name twice, twice as often)
              --shapes PROFILE    the shapes of the shapes layout, again for more, one drawn per page: a name (none:
                                  no shapes, grid: a grid fitted to the text, ruled: a line under each line, boxed: a
                                  box around the text) or one written as text, steps NAME[=VALUE][@CHANCE] with their
                                  options, "grid=auto x 18,widths=1:4:1*16,heavy-every=4,heavy-width=2..3" (shapes:
                                  line, polyline, polygon, box, grid, circle, ellipse, arc, arrow, fill, hatch, text,
                                  ring-text, lettering, star, shield, scribble; any takes flow=down to stack down the
                                  page, lines=N|rest, holds=paragraphs, valign=center|top|bottom|SHARE (where
                                  paragraphs sit); also text-grid-text, framed, circled; none)
              --grey P            the share of pages in grey (0.3)
              --page SETTINGS     page settings, KEY=VALUE,...: width, height, size, spacing, margin, ink, justify,
                                  color, rule, emblem, stamp, blue-signature, header-font, crop (a chance
                                  the page is cut to its text), crop-padding (pixels; ranges as MIN..MAX); for the
                                  shapes layout: pen (ideal, ballpoint), pen-wobble, pen-jitter, pen-pressure,
                                  pen-overshoot, pen-tilt (ranges), bind (sequence), take (units per region, as
                                  1:rest), blank (the chance a region is left empty), blank-at (which, from 1, as
                                  3..18), blue-pen (a chance), cell-align (start, center, end, mixed), stroke (a share
                                  of the text size), least-size (the smallest a region's text is set, a share of size),
                                  sections (blank: empty lines split the text into sections, one a step);
                                  for the letter layout: emblem-shapes, stamp-shapes and signature-shapes (shape
                                  profiles between |, one drawn per page, again for more; emblems: emblem-rings,
                                  emblem-seal, emblem-star, emblem-shield, emblem-badge, emblem-monogram, emblem-box;
                                  stamps: stamp-ring, stamp-seal, stamp-rectangle, stamp-oval, stamp-double;
                                  signature: scribble), emblem-pen and stamp-pen (ideal), signature-pen (ballpoint),
                                  stamp-turn (degrees, -12..12), stamp-ink (how uneven, 0.2..0.65), stamp-black (the
                                  chance a stamp is black; 0.2), signature (the chance a signature is drawn over the
                                  signature block's name; 1);
                                  for every layout: pose (none, drift: each glyph drawn a little differently),
                                  pose-slant, pose-size, pose-baseline, pose-rotate, pose-pressure (ranges); habits
                                  (none, the default; a habits name such as arabic-handwriting, or a rules file: the
                                  body written with a hand's habits), habit-chance (every habit's chance, 0 unless
                                  given), habit.NAME (one habit's), label (drawn, the default: the label says what is
                                  drawn; source: the input's spelling while the image shows the habit)
              --records NAME      the input's record format: sharegpt, messages or text (found from the input)
              --full-text KEY     the answer's key holding the page text (the first full_text..., else the longest text)
              --prompt TEXT       the question of records made from plain text ("Extract the text.")
              --answer-key KEY    write plain text answers as JSON under KEY
              --image-format F    jpeg (the default) or another registered encoder (png)
              --quality Q         the JPEG quality (90; in a PDF, IDRAK_DOCUMENT_JPEG_QUALITY's or 90)
              --format pdf        documents instead of images, as a scanner makes them: each page's image, its look
                                  and all, with its text as a hidden layer on the lines as drawn (a reader finds and
                                  selects it where it shows); the page's size its pixels at --dpi
              --format pdf-vector documents as software makes them: each page as drawn before any look, its text as
                                  glyphs in its fonts (embedded) with the text they stand for, its shapes, rules,
                                  emblems and stamps as paths and their words as text; no look is put on it (--scan
                                  is not used; a pen's wobble and a stamp's uneven ink are written ideal)
              --writer NAME       the document writer (pdf, or another registered one; its extension names the files)
              --pdf-pages N       pages a document (1): a file holds N pages in order, named after its first page;
                                  each record names its document as its image and the page's index in it, from 0, in
                                  "pages"
              --pdf-images E      how a scan-like page's image is stored: auto (JPEG for scanned looks, exact for
                                  line art, measured per page), flate (exact) or jpeg (IDRAK_DOCUMENT_IMAGE_ENCODING;
                                  auto)
              --pdf-fonts E       how the fonts are embedded: subset (the glyphs used) or whole
                                  (IDRAK_DOCUMENT_EMBED_FONTS; subset)
              --dpi N             the drawn pages' pixels an inch in a document, 1 to 9600 (IDRAK_DOCUMENT_DPI; 150:
                                  the default page, 1240 pixels wide, is A4's width)
              --zip               the images (or documents) in a zip instead of a folder
              --workers N         pages made at once (one per processor)
              --limit N           read at most N records (a trial)
              --seed N            the data set's seed (0): the same seed and inputs give the same bytes

        Examples:
          idrak data synth train.jsonl -o synthetic/pages.jsonl --fonts fonts --parts parts.json -n 3
          idrak synth letters.txt -o pages.jsonl --fonts fonts --answer-key full_text --scan old --scan office
          idrak data synth train.jsonl -o trial.jsonl --fonts fonts --limit 5 --page "size=20..28,stamp=0.5" --zip
          idrak data synth rows.txt -o rows.jsonl --fonts fonts --layout shapes --shapes grid --page pen=ballpoint
          idrak data synth letters.txt -o letters.jsonl --fonts fonts --parts parts.json --layout letter
            --page "emblem=1,stamp=0.6,stamp-shapes=stamp-seal|stamp-oval"
          idrak data synth letters.txt -o scans/pages.jsonl --fonts fonts --format pdf --scan office
          idrak data synth letters.txt -o vectors/pages.jsonl --fonts fonts --format pdf-vector --pdf-pages 10
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } =
        ["--out", "--fonts", "--parts", "--count", "--layout", "--scan", "--shapes", "--grey", "--page", "--records", "--full-text", "--prompt", "--answer-key",
         "--image-format", "--quality", "--workers", "--limit", "--writer", "--pdf-pages", "--pdf-images", "--pdf-fonts", "--dpi", .. FontMapSettings.Options];

    // The documents --format names, over the common output formats; the option names that apply to them alone.
    private static readonly string[] PdfFormats = ["pdf", "pdf-vector"];
    private static readonly string[] PdfOptions = ["--writer", "--pdf-pages", "--pdf-images", "--pdf-fonts", "--dpi"];

    public override IReadOnlyCollection<string> OwnFormats => PdfFormats;

    public override IReadOnlyCollection<string> Flags { get; } = ["--zip"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-o"] = "--out", ["-n"] = "--count" };

    public override int Run(CommandContext context)
    {
        string input = context.Argument(0, "INPUT");
        string output = context.Option("--out") ?? throw new UsageException("Give -o, --out FILE for the records to write (.jsonl).");
        string fontsFolder = context.Option("--fonts") ?? throw new UsageException($"Give --fonts DIR, a folder of fonts ({FontMapSettings.FontFiles()}).");
        if (!File.Exists(input))
        {
            throw new UsageException($"The records file {input} is not there (idrak data text writes one).");
        }

        if (!Directory.Exists(fontsFolder))
        {
            throw new UsageException($"The fonts folder {fontsFolder} is not there.");
        }

        using var fontMaps = FontMapSettings.Apply(context);
        var fonts = SyntheticPages.Fonts(fontsFolder);
        if (fonts.Count == 0)
        {
            throw new UsageException($"No font was read from {fontsFolder} ({FontMapSettings.FontFiles()}).");
        }

        var pageOutput = PageOutput(context);
        SyntheticPageOptions options;
        try
        {
            var defaults = new SyntheticPageOptions { Style = new PageStyle { Fonts = fonts } };
            var layouts = context.Options("--layout").SelectMany(l => l.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)).ToList();

            // A profile written as text holds commas; names may be given as a list.
            var scans = context.Options("--scan").SelectMany(s => s.Contains('=') ? [s] : s.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)).ToList();
            var shapes = context.Options("--shapes").SelectMany(s => s.Contains('=') ? [s] : s.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)).ToList();
            options = defaults with
            {
                Style = SyntheticPages.Style(defaults.Style, context.Option("--page")),
                Parts = context.Option("--parts") is { } parts ? SyntheticPageParts.Read(parts) : SyntheticPageParts.Empty,
                Variants = context.IntOption("--count", defaults.Variants),
                Layouts = layouts.Count > 0 ? layouts : defaults.Layouts,
                Scans = scans.Count > 0 ? scans : defaults.Scans,
                Shapes = shapes.Count > 0 ? shapes : defaults.Shapes,
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
                Output = pageOutput,
                Writer = context.Option("--writer") ?? defaults.Writer,
                PagesPerDocument = context.IntOption("--pdf-pages", defaults.PagesPerDocument),
                Dpi = PageDpi(context, defaults.Dpi),
                WriteOptions = PdfWriteOptions(context, defaults.WriteOptions),
            };
            if (pageOutput != SyntheticPageOutput.Images)
            {
                _ = NotOnTheNewDesign.Yet<IDocumentWriter>($"Picking the document writer '{options.Writer}' by its name");
                if (options.PagesPerDocument < 1)
                {
                    throw new UsageException($"--pdf-pages needs a number of pages from 1, not {options.PagesPerDocument}.");
                }
            }
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
            $"{result.Pages:N0} pages from {result.Records:N0} records in {result.Elapsed.TotalSeconds:F1} s: {result.Data}, {(pageOutput == SyntheticPageOutput.Images ? "images" : "documents")} in {result.Images}, manifest {result.Manifest}"));
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
            ["format"] = pageOutput switch { SyntheticPageOutput.ScannedDocuments => "pdf", SyntheticPageOutput.VectorDocuments => "pdf-vector", _ => options.ImageFormat },
            ["seconds"] = Math.Round(result.Elapsed.TotalSeconds, 2),
            ["skipped"] = new JsonArray([.. result.Skips.Select(s => (JsonNode)s)]),
        });
        return ExitCodes.Ok;
    }

    // What the pages are written as: --format pdf or pdf-vector (documents), else images; the document options given with
    // images are refused, naming them.
    private static SyntheticPageOutput PageOutput(CommandContext context)
    {
        var output = context.Option("--format") switch
        {
            "pdf" => SyntheticPageOutput.ScannedDocuments,
            "pdf-vector" => SyntheticPageOutput.VectorDocuments,
            _ => SyntheticPageOutput.Images,
        };
        if (output == SyntheticPageOutput.Images && PdfOptions.Where(o => context.Option(o) is not null).ToList() is { Count: > 0 } given)
        {
            throw new UsageException($"{string.Join(", ", given)} {(given.Count == 1 ? "applies" : "apply")} to documents: give --format pdf or pdf-vector with {(given.Count == 1 ? "it" : "them")}.");
        }

        if (output == SyntheticPageOutput.VectorDocuments && context.Option("--pdf-images") is not null)
        {
            throw new UsageException("--pdf-images stores a scan-like page's image (--format pdf): a pdf-vector page has none.");
        }

        return output;
    }

    // --dpi: the drawn pages' pixels an inch in a document (unless given, 0: IDRAK_DOCUMENT_DPI's, else 150).
    private static float PageDpi(CommandContext context, float fallback)
    {
        if (context.Option("--dpi") is not { } text)
        {
            return fallback;
        }

        return float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out float dpi) && dpi is >= 1f and <= 9600f
            ? dpi : throw new UsageException($"--dpi takes a resolution from 1 to 9600 pixels an inch, not '{text}'.");
    }

    // The document writer's options: --pdf-images, --pdf-fonts and --quality over the writer's variables.
    private static DocumentWriteOptions PdfWriteOptions(CommandContext context, DocumentWriteOptions defaults)
    {
        var options = defaults;
        if (context.Option("--pdf-images") is { } encoding)
        {
            options = options with
            {
                ImageEncoding = encoding.ToLowerInvariant() switch
                {
                    "auto" => DocumentImageEncoding.Auto,
                    "flate" => DocumentImageEncoding.Flate,
                    "jpeg" => DocumentImageEncoding.Jpeg,
                    _ => throw new UsageException($"--pdf-images takes auto, flate or jpeg, not '{encoding}'."),
                },
            };
        }

        if (context.Option("--pdf-fonts") is { } fonts)
        {
            options = options with
            {
                EmbedFonts = fonts.ToLowerInvariant() switch
                {
                    "subset" => DocumentFontEmbedding.Subset,
                    "whole" => DocumentFontEmbedding.Whole,
                    _ => throw new UsageException($"--pdf-fonts takes subset or whole, not '{fonts}'."),
                },
            };
        }

        if (context.Option("--quality") is not null)
        {
            int quality = context.IntOption("--quality", defaults.JpegQuality);
            options = options with { JpegQuality = quality is >= 1 and <= 100 ? quality : throw new UsageException($"--quality takes 1 to 100, not {quality}.") };
        }

        return options;
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
