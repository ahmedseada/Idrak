// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Gemma3Vision;

namespace Idrak.Samples.ArabicOcr;

/// <summary>What a command runs with: its arguments, where it writes, the device, and whether it answers in JSON.</summary>
internal sealed class OcrContext(Arguments arguments, TextWriter output, TextWriter error)
{
    public Arguments Args { get; } = arguments;

    public TextWriter Output { get; } = output;

    public TextWriter Error { get; } = error;

    public bool Json => Args.Flag("--json");

    public bool Verbose => Args.Flag("--verbose");

    /// <summary>The device of <c>--device</c> (auto, cpu, cuda, vulkan, cuda:N, vulkan:N): the library's default when not given.</summary>
    public Device Device { get; } = OcrApp.ParseDevice(arguments.Option("--device"));

    /// <summary>A line of progress (to the error stream, so the output stays the result).</summary>
    public void Say(string line)
    {
        Error.WriteLine(line);
        Error.Flush();
    }

    /// <summary>Prints a JSON document (indented, Arabic as itself).</summary>
    public void WriteJson(JsonNode json) => Output.WriteLine(json.ToJsonString(ArabicOcr.Json.Indented));
}

/// <summary>The app: one console program with a command per task (read, cut, accept, train, eval, tune-vlm).</summary>
internal static class OcrApp
{
    public const string Usage = """
        Idrak OCR: text from page images (Arabic first; any script).

          read <image|folder|data.json>... --reader lines|vlm [-o FILE|DIR] [-j]
               lines: page -> lines -> the trained line recognizer (--model DIR, --decoder greedy|beam, --beam-width N,
                      --single-line: each image is one line already)
               vlm:   a Gemma 3 vision checkpoint (--model DIR, --adapter DIR, --image-transform PIPELINE, --grayscale,
                      --pan-and-scan, --vision-option KEY=VALUE, --prompt TEXT, --system TEXT, --max-tokens N)
          cut <image|folder|data.json>... -o DIR [--prefill vlm --model DIR ...] [--truth data.json [--images DIR|ZIP]]
               each line as NAME-line-NN.png with an empty NAME-line-NN.txt to fill in; with --prefill vlm a draft instead
               (NAME-line-NN.draft.txt), snapped to the page's known text when there is one (NAME-line-NN.aligned.txt).
               Never overwrites a filled .txt.
          accept DIR [--drafts]
               renames each .aligned.txt (with --drafts also each .draft.txt) to .txt where no filled .txt exists
          train --data DIR [--eval DIR] [-o MODEL] [--epochs N] [--batch N] [--height N] [--include-drafts] [--augment PIPELINE]
          eval <image|folder|data.json>... --reader lines|vlm [--truth DIR|FILE] [--truth-text raw|values] [-j]
          tune-vlm --model DIR --data train.json [--eval val.json] [--images DIR|ZIP] [-o ADAPTER]
               prints the idrak tune command (plan 12) for that data

        Common: --device auto|cpu|cuda|vulkan|cuda:N|vulkan:N, -j/--json, -v/--verbose, -h/--help.
        Data files (.json, .jsonl: ShareGPT or chat messages) take --images DIR|ZIP for their images.
        """;

    /// <summary>Runs the app with <paramref name="args"/>; returns the exit code (0 done, 1 failed, 2 usage).</summary>
    public static int Run(string[] args, TextWriter output, TextWriter error)
    {
        if (args.Length == 0 || args[0] is "-h" or "--help" or "help")
        {
            output.WriteLine(Usage);
            return args.Length == 0 ? 2 : 0;
        }

        // The app brings the families and augmentations it uses: the registries know nothing of them otherwise.
        Gemma3VisionPlugin.Register();
        LineNoise.Register();

        string command = args[0];
        var rest = args.Skip(1).ToArray();
        try
        {
            return command switch
            {
                "read" => ReadCommand.Run(rest, output, error),
                "cut" => CutCommand.Run(rest, output, error),
                "accept" => CutCommand.Accept(rest, output, error),
                "train" => TrainCommand.Run(rest, output, error),
                "eval" => EvalCommand.Run(rest, output, error),
                "tune-vlm" => TuneVlmCommand.Run(rest, output, error),
                _ => throw new UsageException($"Unknown command '{command}'.\n{Usage}"),
            };
        }
        catch (UsageException e)
        {
            error.WriteLine($"error: {e.Message}");
            return 2;
        }
        catch (Exception e) when (e is InvalidDataException or IOException or NotSupportedException or InvalidOperationException or ArgumentException or UnauthorizedAccessException)
        {
            error.WriteLine($"error: {e.Message}");
            return 1;
        }
    }

    /// <summary>Options every command takes.</summary>
    public static readonly string[] CommonValues = ["--device"];

    /// <summary>Flags every command takes.</summary>
    public static readonly string[] CommonFlags = ["--json", "--verbose", "--help"];

    /// <summary>The context of a command that takes <paramref name="values"/> and <paramref name="flags"/> besides the common ones; null after printing help.</summary>
    public static OcrContext? Context(string[] args, TextWriter output, TextWriter error, string[] values, string[] flags, string help)
    {
        var parsed = Arguments.Parse(args, [.. CommonValues, .. values], [.. CommonFlags, .. flags]);
        if (parsed.Flag("--help"))
        {
            output.WriteLine(help);
            return null;
        }

        return new OcrContext(parsed, output, error);
    }

    public static Device ParseDevice(string? text) => text?.ToLowerInvariant() switch
    {
        null or "auto" => Device.Default,
        "cuda" or "gpu" => Device.Cuda(),
        "vulkan" => Device.Parse("vulkan:0"),
        var name => Device.Parse(name),
    };
}
