// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Cli.Shared;

namespace Idrak.Cli.Commands;

/// <summary>
/// <c>idrak convert IN OUT</c>: writes a model read from a GGUF file or a Hugging Face folder as a Hugging Face folder
/// (safetensors in bf16, f16 or f32), e.g. to use a GGUF model with other tools or to change a folder's element type.
/// </summary>
internal sealed class ConvertCommand : Command
{
    public override string Name => "convert";

    public override string Summary => "Convert between GGUF and Hugging Face folders (safetensors bf16, f16, f32)";

    public override string Usage => """
        IN OUT [--type bf16|f16|f32] [-f] [--dry-run]

          IN              a .gguf file, a model folder, a cached Hugging Face id or an alias
          OUT             a folder: the model in the Hugging Face layout (model.safetensors, config, tokenizer, template)
          -o, --out DIR   OUT as an option
              --type T    the element type of the written weights (default bf16)
          -f, --force     write into a folder that is not empty
              --dry-run   say what would be written

        GGUF weights are dequantized (float32 in memory, one model at a time) and written in --type.

        Gaps: writing GGUF files (the library reads GGUF but has no writer) and language models as .ikm packages (a
        package holds a network built from a builder description; a pretrained decoder is not loaded from one yet).

        Examples:
          idrak convert ./qwen3-0.6b-q8_0.gguf ./qwen3-0.6b
          idrak convert Qwen/Qwen3-0.6B -o ./qwen-f32 --type f32
        """;

    public override IReadOnlyCollection<string> ValueOptions => ["--out", "--type"];

    public override IReadOnlyCollection<string> Flags => ["--force", "--dry-run"];

    public override IReadOnlyDictionary<string, string> ShortForms => new Dictionary<string, string> { ["-o"] = "--out", ["-f"] = "--force" };

    public override int Run(CommandContext context)
    {
        string input = context.Argument(0, "IN (a .gguf file or a model folder)");
        string output = context.Option("--out") ?? (context.Positional.Count > 1 ? context.Positional[1] : throw new UsageException("Missing OUT (the folder to write)."));
        if (output.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException("Writing GGUF files is not supported yet (the library reads GGUF but has no writer); convert to a folder instead.");
        }

        if (output.EndsWith(".ikm", StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException("Writing a language model as an .ikm package is not supported yet (packages hold builder-described networks); convert to a folder instead.");
        }

        var type = ModelWork.Type(context);
        ModelWork.CheckOutput(context, output);
        var local = ModelCache.Locate(context, input);
        string from = local.File is null ? "Hugging Face folder" : "GGUF file";
        if (context.Flag("--dry-run"))
        {
            context.Write($"Would read {input} ({from}) and write it to {output} as a Hugging Face folder ({type}).");
            context.WriteJson(new JsonObject { ["in"] = input, ["from"] = local.File is null ? "huggingface" : "gguf", ["out"] = output, ["type"] = type.ToString(), ["dryRun"] = true });
            return ExitCodes.Ok;
        }

        var clock = System.Diagnostics.Stopwatch.StartNew();
        using (var model = Shared.Models.Load(context, new Shared.Models.ModelChoice(local.Folder, null, null, null, null)))
        {
            ModelWork.SaveHuggingFace(model, output, type);
        }

        long bytes = ModelCache.FolderBytes(output);
        context.Write($"Converted {input} ({from}) to {output}: {ModelCache.Size(bytes)}, {type}, in {clock.Elapsed.TotalSeconds:F1} s.");
        context.WriteJson(new JsonObject
        {
            ["in"] = input,
            ["from"] = local.File is null ? "huggingface" : "gguf",
            ["out"] = output,
            ["type"] = type.ToString(),
            ["bytes"] = bytes,
            ["files"] = new JsonArray([.. Directory.GetFiles(output).Select(Path.GetFileName).Order(StringComparer.Ordinal).Select(f => (JsonNode?)JsonValue.Create(f))]),
        });
        return ExitCodes.Ok;
    }
}
