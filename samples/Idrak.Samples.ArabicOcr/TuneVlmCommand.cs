// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Samples.ArabicOcr;

/// <summary>
/// <c>tune-vlm</c>: a pointer, not a tuner. Fine-tuning the vision-language model is the library's (<c>idrak tune</c>,
/// plan 12); this prints the exact commands for the given data, so the adapter it writes is the one <c>read --reader vlm
/// --adapter</c> loads.
/// </summary>
internal static class TuneVlmCommand
{
    private const string Help = """
        tune-vlm --model DIR --data train.json [--eval val.json] [--images DIR|ZIP] [-o ADAPTER] [--epochs N] [--grayscale]
          Prints the PowerShell commands that fine-tune the model with idrak tune (plan 12: LoRA, the Gemma 3 plug-in, CER on
          the evaluation set) and score it before and after; nothing is run.
        """;

    public static int Run(string[] args, TextWriter output, TextWriter error)
    {
        if (OcrApp.Context(args, output, error, ["--model", "--data", "--eval", "--images", "--out", "--epochs", "--image-transform"], ["--grayscale"], Help) is not { } context)
        {
            return 0;
        }

        var a = context.Args;
        string model = Full(a.Required("--model", "the Gemma 3 vision checkpoint to tune"));
        string data = Full(a.Required("--data", "the training conversations (ShareGPT or messages)"));
        string? eval = a.Option("--eval") is { } e ? Full(e) : null;
        string? images = a.Option("--images") is { } i ? Full(i) : null;
        string adapter = a.Option("--out") is { } o ? Full(o) : model.TrimEnd('\\', '/') + "-lora";
        int epochs = a.Integer("--epochs", 1, 1);
        string root = RepositoryRoot() ?? @"D:\Projects\Idrak";
        string plugin = Path.Combine(root, "samples", "Gemma3Vision", "Idrak.Gemma3Vision", "bin", "Release", "net10.0", "Idrak.Gemma3Vision.dll");
        string prep = (a.Flag("--grayscale") ? " --grayscale" : "") + (a.Option("--image-transform") is { } t ? $" --image-transform \"{t}\"" : "");
        string imagesOption = images is null ? "" : $" --images \"{images}\"";
        string cli = $"dotnet run -c Release --project \"{Path.Combine(root, "src", "Idrak.Cli")}\" --";
        var lines = new List<string>
        {
            "# Fine-tuning is idrak tune's (plan 12); these are its commands for this data. From PowerShell:",
            $"cd \"{root}\"",
            $"dotnet build -c Release \"{Path.Combine(root, "samples", "Gemma3Vision", "Idrak.Gemma3Vision")}\"",
            $"$plugin = \"{plugin}\"",
        };
        if (eval is not null)
        {
            lines.Add($"{cli} tune evaluate -P $plugin \"{model}\" \"{eval}\"{imagesOption}{prep} --metric cer --samples 20 -v");
        }

        lines.Add($"{cli} tune -P $plugin -b \"{model}\" --data \"{data}\"{(eval is null ? " --eval-fraction 0.05" : $" --eval \"{eval}\"")}{imagesOption}{prep} "
                  + $"--metric cer --metric-every 50 --metric-samples 20 --epochs {epochs} -o \"{adapter}\" -v");
        if (eval is not null)
        {
            lines.Add($"{cli} tune evaluate -P $plugin \"{model}\" \"{eval}\"{imagesOption} --adapter \"{adapter}\" --metric cer --samples 20");
        }

        lines.Add("# Then read with the adapter (its image preparation comes with it):");
        lines.Add($"dotnet run -c Release --project \"{Path.Combine(root, "samples", "Idrak.Samples.ArabicOcr")}\" -- read SCAN.png --reader vlm --model \"{model}\" --adapter \"{adapter}\"");
        foreach (string line in lines)
        {
            context.Output.WriteLine(line);
        }

        return 0;
    }

    private static string Full(string path) => Path.GetFullPath(path);

    // The clone this app runs from (the folder holding Idrak.slnx), or null.
    private static string? RepositoryRoot()
    {
        for (var folder = new DirectoryInfo(AppContext.BaseDirectory); folder is not null; folder = folder.Parent)
        {
            if (File.Exists(Path.Combine(folder.FullName, "Idrak.slnx")))
            {
                return folder.FullName;
            }
        }

        return null;
    }
}
