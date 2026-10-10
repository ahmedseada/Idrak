// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Samples.ArabicOcr;

/// <summary>
/// <c>cut</c>: each page's lines as images to transcribe, and <c>accept</c>: drafts taken as corrected. The files are
/// <see cref="LineFiles"/>' layout. Idempotent: the same page gives the same line images again; a filled <c>.txt</c> is never
/// touched, and an existing draft is kept.
/// </summary>
internal static class CutCommand
{
    private const string Help = """
        cut <image|folder|data.json>... -o DIR
          Writes each detected line as DIR/NAME-line-NN.png (NAME: the scan's file name, or DATA-NNNN for record NNNN of a
          data file; lines numbered top to bottom) and beside it NAME-line-NN.txt, empty, to fill in. Never overwrites a
          filled .txt.
          --prefill vlm --model DIR   instead of an empty .txt, a draft: the vision-language reader's reading of the line
                                      (NAME-line-NN.draft.txt); with the reader's options (--adapter, --image-transform,
                                      --grayscale, --pan-and-scan, --vision-option, --prompt, --max-tokens)
          --truth data.json           the pages' known text (ShareGPT or messages; a scan finds its page by its image's bytes;
                                      a data file given as input is its own truth): each draft is snapped to the best-matching
                                      span of its page's text, top to bottom, and written as NAME-line-NN.aligned.txt
          --truth-text raw|values     the truth as it is, or the text values of a JSON answer, in order (values)
          --images DIR|ZIP            where a data file's images are (a zip is read in place)
          --max-skew DEGREES          the largest skew corrected (3; 0: none)
        Accept a draft by renaming it to NAME-line-NN.txt (after correcting it), or with: accept DIR [--drafts]
        """;

    private const string AcceptHelp = """
        accept DIR [--drafts]
          Renames every NAME-line-NN.aligned.txt (with --drafts also every .draft.txt without an aligned one) to
          NAME-line-NN.txt where no filled .txt exists (an empty one is replaced). A filled .txt is never touched.
        """;

    public static int Run(string[] args, TextWriter output, TextWriter error)
    {
        if (OcrApp.Context(args, output, error,
                ["--out", "--prefill", "--model", "--truth", "--truth-text", "--images", "--max-skew", .. Readers.VlmValues],
                [.. Readers.VlmFlags], Help) is not { } context)
        {
            return 0;
        }

        var a = context.Args;
        string folder = a.Required("--out", "the folder for the line images");
        string? prefill = a.Option("--prefill");
        if (prefill is not (null or Readers.Vlm))
        {
            throw new UsageException($"--prefill takes {Readers.Vlm} (the vision-language reader), not '{prefill}'.");
        }

        string? truthFile = a.Option("--truth");
        string truthMode = a.Option("--truth-text") ?? "values";
        _ = OcrText.AnswerText("", truthMode);                                         // checks the mode
        string? images = a.Option("--images");
        var truths = truthFile is null ? null : Pages.ByImage(truthFile, images);
        var segmentation = new SegmentationSettings { MaxSkew = a.Number("--max-skew", 3, 0) };
        using var reader = prefill is null ? null : Readers.Create(Readers.Vlm, context);
        Directory.CreateDirectory(folder);

        var report = new JsonArray();
        int linesTotal = 0, created = 0, drafted = 0, aligned = 0, kept = 0;
        foreach (var page in Pages.Read(a.Words, images))
        {
            string? truth = page.Truth ?? (truths is not null && truths.TryGetValue(page.Image.Hash, out var known) ? known.Truth : null);
            var (lines, skew) = LineSegmenter.Find(page.Decode(), segmentation);
            var files = new List<string>();
            foreach (var line in lines)
            {
                string image = Path.Combine(folder, $"{page.Name}-line-{line.Index:D2}.png");
                ImageEncoders.Save(image, line.Image);
                files.Add(image);
            }

            // What each line already has (its corrected text, else its draft), and the lines that need a reading.
            var texts = files.Select(f => LineFiles.IsCorrected(f) ? OcrText.ReadTranscription(LineFiles.Of(f, LineFiles.Corrected))
                : FirstFilled(OcrText.ReadTranscription(LineFiles.Of(f, LineFiles.Aligned)), OcrText.ReadTranscription(LineFiles.Of(f, LineFiles.Draft)))).ToArray();
            int pageKept = files.Count(LineFiles.IsCorrected);
            var fresh = new List<int>();
            for (int i = 0; i < files.Count; i++)
            {
                if (texts[i].Length > 0)
                {
                    continue;
                }

                if (reader is null)
                {
                    string txt = LineFiles.Of(files[i], LineFiles.Corrected);
                    if (!File.Exists(txt))
                    {
                        OcrText.Write(txt, "");
                        created++;
                    }

                    continue;
                }

                var reading = reader.Read(new Page(Path.GetFileNameWithoutExtension(files[i]), files[i], ChatImage.FromFile(files[i])));
                texts[i] = OcrText.Normalize(OcrText.AnswerText(reading.Text, "values"));
                fresh.Add(i);
            }

            // Snapped to the page's known text: every line's text (corrected ones too) anchors the order.
            var snapped = truth is null || fresh.Count == 0 ? null : TextAlignment.Align(texts, OcrText.Normalize(OcrText.AnswerText(truth, truthMode)));
            int pageAligned = 0;
            foreach (int i in fresh)
            {
                if (snapped?[i] is { } match)
                {
                    OcrText.Write(LineFiles.Of(files[i], LineFiles.Aligned), match.Text + "\n");
                    pageAligned++;
                }
                else
                {
                    OcrText.Write(LineFiles.Of(files[i], LineFiles.Draft), texts[i] + "\n");
                }
            }

            linesTotal += files.Count;
            kept += pageKept;
            aligned += pageAligned;
            drafted += fresh.Count - pageAligned;
            string summary = $"{page.Source}: {files.Count} lines"
                             + (skew != 0 ? FormattableString.Invariant($" (deskewed {skew:F1}°)") : "")
                             + (pageKept > 0 ? $", {pageKept} corrected kept" : "")
                             + (fresh.Count > 0 ? $", {fresh.Count} drafts ({pageAligned} aligned to the page's text)" : "")
                             + (truth is null && reader is not null && truthFile is not null ? ", no known text for this page" : "");
            context.Say(summary);
            report.Add(new JsonObject
            {
                ["name"] = page.Name, ["source"] = page.Source, ["lines"] = files.Count, ["skew_degrees"] = skew, ["corrected_kept"] = pageKept,
                ["drafts"] = fresh.Count, ["aligned"] = pageAligned, ["truth"] = truth is not null,
            });
        }

        context.Say($"{linesTotal} lines in {Path.GetFullPath(folder)}: {kept} corrected kept, {created} new empty .txt, {aligned} aligned drafts, {drafted} drafts");
        if (context.Json)
        {
            context.WriteJson(new JsonObject
            {
                ["folder"] = Path.GetFullPath(folder), ["lines"] = linesTotal, ["corrected_kept"] = kept, ["created"] = created,
                ["aligned"] = aligned, ["drafts"] = drafted, ["pages"] = report,
            });
        }

        return 0;
    }

    /// <summary><c>accept DIR [--drafts]</c>.</summary>
    public static int Accept(string[] args, TextWriter output, TextWriter error)
    {
        if (OcrApp.Context(args, output, error, [], ["--drafts"], AcceptHelp) is not { } context)
        {
            return 0;
        }

        string folder = context.Args.Words.Count == 1 ? context.Args.Words[0] : throw new UsageException("accept takes one folder (cut's).");
        int accepted = 0, kept = 0;
        foreach (string image in Pages.ImageFiles(folder, recursive: true))
        {
            string txt = LineFiles.Of(image, LineFiles.Corrected);
            if (LineFiles.IsCorrected(image))
            {
                kept++;
                continue;
            }

            foreach (string kind in context.Args.Flag("--drafts") ? new[] { LineFiles.Aligned, LineFiles.Draft } : [LineFiles.Aligned])
            {
                string draft = LineFiles.Of(image, kind);
                if (OcrText.ReadTranscription(draft).Length > 0)
                {
                    File.Move(draft, txt, overwrite: true);                                // an empty .txt only: a filled one was kept above
                    accepted++;
                    break;
                }
            }
        }

        context.Say($"{accepted} drafts accepted as .txt, {kept} corrected .txt kept");
        if (context.Json)
        {
            context.WriteJson(new JsonObject { ["accepted"] = accepted, ["kept"] = kept });
        }

        return 0;
    }

    private static string FirstFilled(string a, string b) => a.Length > 0 ? a : b;
}
