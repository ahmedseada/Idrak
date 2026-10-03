// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.LanguageModels;

namespace Idrak.Cli.Commands;

/// <summary>
/// <c>idrak quantize MODEL -w FORMAT</c>: packs the model's weights in a registered packed format on the chosen device,
/// reports the size against float32 and a quick quality check (perplexity of a short text in both), and with
/// <c>--out DIR</c> writes the packed weights.
/// </summary>
internal sealed class QuantizeCommand : Command
{
    public override string Name => "quantize";

    public override string Summary => "Pack a model's weights (int8, int4, bf16 or any registered format): size and a quick perplexity check";

    public override string Usage => """
        MODEL -w FORMAT [-o DIR] [--text FILE] [--tokens N] [-f] [--dry-run]

        Arguments:
          MODEL  a cached Hugging Face id, a folder, a .gguf file or an alias

        Options:
          -w, --weights F  int8, int4, bf16 or any registered packed format (plug-ins add more with --plugin)
          -o, --out DIR    write the packed weights (weights.bin), the config and tokenizer files, and quantize.json
              --text FILE  the text the perplexity is measured on (default a short built-in paragraph)
              --tokens N   at most N tokens of it (default 256)
          -f, --force      write into a folder that is not empty
              --dry-run    say what would be done without loading the model

        Examples:
          idrak quantize Qwen/Qwen3-0.6B -w int4
          idrak quantize ./model -w int8 -o ./model-int8 -d vulkan:0

        Gap: the library packs weights while it loads a checkpoint and has no loader for packed weights yet, so
        chat, run and serve pack at load time (-w FORMAT gives the same weights); weights.bin is the packed module
        as Module.Save writes it, for Module.Load into a model loaded with the same format.
        """;

    public override IReadOnlyCollection<string> ValueOptions => ["--weights", "--out", "--text", "--tokens"];

    public override IReadOnlyCollection<string> Flags => ["--force", "--dry-run"];

    public override IReadOnlyDictionary<string, string> ShortForms => new Dictionary<string, string> { ["-w"] = "--weights", ["-o"] = "--out", ["-f"] = "--force" };

    public override int Run(CommandContext context)
    {
        string name = context.Argument(0, "MODEL");
        string format = context.Option("--weights") ?? Shared.Models.Choose(context, name).Weights
            ?? throw new UsageException("Missing -w FORMAT (int8, int4, bf16 or a registered packed format).");
        _ = MemoryCommand.BytesPerValue(format);                                  // an unknown format is a usage error before loading
        string? output = context.Option("--out");
        if (output is not null)
        {
            ModelWork.CheckOutput(context, output);
        }

        var local = ModelCache.Locate(context, name);
        if (Terminal.DryRun(context))
        {
            context.Write($"Would load {name} ({local.File ?? local.Folder}) on {context.Device} in float32 and in {format}, compare their size and perplexity"
                          + (output is null ? "." : $", and write the {format} weights to {output}."));
            context.WriteJson(new JsonObject { ["model"] = name, ["weights"] = format, ["out"] = output, ["dryRun"] = true });
            return ExitCodes.Ok;
        }

        int limit = context.IntOption("--tokens", 256);
        long floatBytes;
        double floatPerplexity;
        int[] tokens;
        using (var reference = Shared.Models.Load(context, new Shared.Models.ModelChoice(local.Folder, null, null, null, null)))
        {
            tokens = ModelWork.CheckTokens(context, reference, limit);
            floatBytes = ModelWork.Bytes(reference.Network);
            floatPerplexity = ModelWork.Perplexity(reference, tokens);
        }

        using var packed = Shared.Models.Load(context, new Shared.Models.ModelChoice(local.Folder, format, null, null, null));
        long packedBytes = ModelWork.Bytes(packed.Network);
        double packedPerplexity = ModelWork.Perplexity(packed, tokens);
        double change = (packedPerplexity / floatPerplexity - 1) * 100;

        context.Write($"{name} in {format} on {context.Device}");
        context.Table(["Weights", "Size", "Perplexity"],
        [
            ["float32", Units.Bytes(floatBytes), $"{floatPerplexity:F3}"],
            [format, $"{Units.Bytes(packedBytes)} ({packedBytes * 100.0 / floatBytes:F0}%)", $"{packedPerplexity:F3} ({change:+0.00;-0.00}%)"],
        ]);
        context.Write($"Perplexity of {tokens.Length} tokens of {(context.Option("--text") ?? "the built-in text")}: the change is what the format costs (a few percent is usual for 4-bit weights).");

        if (output is not null)
        {
            Directory.CreateDirectory(output);
            packed.Network.Save(Path.Combine(output, "weights.bin"));
            foreach (string file in Directory.GetFiles(local.Folder))
            {
                string fileName = Path.GetFileName(file);
                if ((fileName.EndsWith(".json", StringComparison.OrdinalIgnoreCase) || fileName.EndsWith(".jinja", StringComparison.OrdinalIgnoreCase))
                    && fileName is not ("gguf.json" or "model.safetensors.index.json"))
                {
                    File.Copy(file, Path.Combine(output, fileName), overwrite: true);
                }
            }

            File.WriteAllText(Path.Combine(output, "quantize.json"), new JsonObject
            {
                ["source"] = local.File ?? local.Folder,
                ["weights"] = format,
                ["bytes"] = packedBytes,
                ["perplexity"] = packedPerplexity,
                ["floatPerplexity"] = floatPerplexity,
                ["tokens"] = tokens.Length,
            }.ToJsonString(CommandContext.JsonOutput));
            context.Write($"Wrote {output} (weights.bin, config and tokenizer files, quantize.json).");
        }

        context.WriteJson(new JsonObject
        {
            ["model"] = name,
            ["device"] = context.Device.ToString(),
            ["weights"] = format,
            ["floatBytes"] = floatBytes,
            ["bytes"] = packedBytes,
            ["floatPerplexity"] = Math.Round(floatPerplexity, 4),
            ["perplexity"] = Math.Round(packedPerplexity, 4),
            ["perplexityChangePercent"] = Math.Round(change, 3),
            ["tokens"] = tokens.Length,
            ["out"] = output,
        });
        return ExitCodes.Ok;
    }
}
