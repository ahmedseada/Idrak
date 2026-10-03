// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Cli.Shared;

namespace Idrak.Cli.Commands;

/// <summary>
/// <c>idrak merge MODEL ADAPTER --out DIR</c>: folds a LoRA or DoRA adapter (a PEFT folder: adapter_config.json and
/// adapter_model.safetensors, as idrak tune and peft write it) into the base weights and writes a Hugging Face folder.
/// </summary>
internal sealed class MergeCommand : Command
{
    public override string Name => "merge";

    public override string Summary => "Merge a LoRA or DoRA adapter into the base weights";

    public override string Usage => """
        MODEL ADAPTER -o DIR [--type bf16|f16|f32] [-f] [--dry-run]

        Arguments:
          MODEL    the base model: a cached Hugging Face id, a folder, a .gguf file or an alias
          ADAPTER  the adapter folder (adapter_config.json, adapter_model.safetensors)

        Options:
          -o, --out DIR  the merged model, in the Hugging Face layout (model.safetensors, config, tokenizer)
              --type T   the element type of the written weights (default bf16)
          -f, --force    write into a folder that is not empty
              --dry-run  check the adapter and say what would be written

        The merge runs in float32 on the chosen device (-d); the merged model loads like any other (idrak chat DIR).

        Examples:
          idrak merge Qwen/Qwen3-0.6B ./run/adapter -o ./qwen-tuned
          idrak merge ./base ./adapter -o ./merged --type f32 -d cpu
        """;

    public override IReadOnlyCollection<string> ValueOptions => ["--out", "--type"];

    public override IReadOnlyCollection<string> Flags => ["--force", "--dry-run"];

    public override IReadOnlyDictionary<string, string> ShortForms => new Dictionary<string, string> { ["-o"] = "--out", ["-f"] = "--force" };

    public override int Run(CommandContext context)
    {
        string name = context.Argument(0, "MODEL"), adapter = context.Argument(1, "ADAPTER (a folder with adapter_config.json)");
        string output = context.Option("--out") ?? (context.Positional.Count > 2 ? context.Positional[2] : throw new UsageException("Missing -o DIR (where the merged model goes)."));
        var type = ModelWork.Type(context);
        if (!File.Exists(Path.Combine(adapter, "adapter_config.json")) || !File.Exists(Path.Combine(adapter, "adapter_model.safetensors")))
        {
            throw new UsageException($"{adapter} is not an adapter folder: it needs adapter_config.json and adapter_model.safetensors.");
        }

        ModelWork.CheckOutput(context, output);
        var local = ModelCache.Locate(context, name);
        var config = JsonNode.Parse(File.ReadAllText(Path.Combine(adapter, "adapter_config.json")));
        string kind = (bool?)config?["use_dora"] == true ? "DoRA" : "LoRA";
        if (Terminal.DryRun(context))
        {
            context.Write($"Would merge the {kind} adapter {adapter} (rank {(int?)config?["r"]}) into {name} and write {output} ({type}).");
            context.WriteJson(new JsonObject { ["model"] = name, ["adapter"] = adapter, ["kind"] = kind, ["out"] = output, ["dryRun"] = true });
            return ExitCodes.Ok;
        }

        var clock = System.Diagnostics.Stopwatch.StartNew();
        using var model = Shared.Models.Load(context, new Shared.Models.ModelChoice(local.Folder, null, null, null, adapter));
        ModelWork.SaveHuggingFace(model, output, type);
        string? merged = model.Notes.FirstOrDefault(n => n.StartsWith("adapter ", StringComparison.Ordinal));
        long bytes = ModelCache.FolderBytes(output);
        context.Write($"Merged the {kind} adapter into {name}: {merged ?? "done"}");
        context.Write($"Wrote {output} ({Units.Bytes(bytes)}, {type}) in {clock.Elapsed.TotalSeconds:F1} s. Try it: idrak run {output} \"Hello\"");
        context.WriteJson(new JsonObject
        {
            ["model"] = name,
            ["adapter"] = adapter,
            ["kind"] = kind,
            ["out"] = output,
            ["type"] = type.ToString(),
            ["bytes"] = bytes,
            ["notes"] = new JsonArray([.. model.Notes.Select(n => (JsonNode?)JsonValue.Create(n))]),
        });
        return ExitCodes.Ok;
    }
}
