// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Samples.ArabicOcr;

/// <summary><c>read</c>: the text of page images with either reader, to the console, a file or a folder.</summary>
internal static class ReadCommand
{
    private const string Help = """
        read <image|folder|data.json>... --reader lines|vlm --model DIR [-o FILE|DIR] [-j]
          --reader lines   the trained line recognizer (--model: its folder); --decoder greedy|beam, --beam-width N,
                           --single-line (each image is one line), --max-skew DEGREES (3; 0: no deskew), --batch N (a ceiling;
                           otherwise measured on the device)
          --reader vlm     a Gemma 3 vision checkpoint (--model: its folder); --adapter DIR (idrak tune's, with its image
                           preparation), --image-transform PIPELINE, --grayscale, --pan-and-scan, --vision-option KEY=VALUE,
                           --prompt TEXT (else the data's prompt, else "Extract the text of this image."), --system TEXT,
                           --max-tokens N (2048), --context N (8192); greedy; the text streams as it is generated
          -o FILE|DIR      one image: the text (or with -j the JSON) to FILE; several: NAME.txt (NAME.json) per page in DIR
          --images DIR|ZIP where a data file's images are (default: beside it)
          -j               JSON: per page its text and the reader's details (lines: boxes and scores)
        """;

    public static int Run(string[] args, TextWriter output, TextWriter error)
    {
        if (OcrApp.Context(args, output, error,
                ["--reader", "--model", "--out", "--images", .. Readers.LineValues, .. Readers.VlmValues],
                [.. Readers.LineFlags, .. Readers.VlmFlags], Help) is not { } context)
        {
            return 0;
        }

        var inputs = context.Args.Words;
        string? outPath = context.Args.Option("--out");
        bool onePage = inputs.Count == 1 && File.Exists(inputs[0]) && !Pages.IsDataFile(inputs[0]);
        bool toFile = outPath is not null && onePage && Path.HasExtension(outPath) && !Directory.Exists(outPath);
        using var reader = Readers.Create(context.Args.Required("--reader", "lines or vlm"), context);
        var pages = new JsonArray();
        foreach (var page in Pages.Read(inputs, context.Args.Option("--images")))
        {
            bool stream = !context.Json && outPath is null;
            if (stream && !onePage)
            {
                context.Output.WriteLine($"==> {page.Source} <==");
            }

            var reading = reader.Read(page, stream ? context.Output : null);
            var json = new JsonObject { ["name"] = page.Name, ["source"] = page.Source, ["text"] = reading.Text };
            foreach (var (key, value) in reading.Details)
            {
                json[key] = value?.DeepClone();
            }

            pages.Add(json);
            if (outPath is not null && !toFile)
            {
                string file = Path.Combine(outPath, page.Name + (context.Json ? ".json" : ".txt"));
                OcrText.Write(file, (context.Json ? json.ToJsonString(Json.Indented) : reading.Text) + "\n");
                context.Say($"{page.Source}: {file}");
            }
            else if (toFile)
            {
                OcrText.Write(outPath!, (context.Json ? json.ToJsonString(Json.Indented) : reading.Text) + "\n");
            }
        }

        if (context.Json && outPath is null)
        {
            context.WriteJson(new JsonObject { ["reader"] = reader.Name, ["pages"] = pages });
        }

        return 0;
    }
}
