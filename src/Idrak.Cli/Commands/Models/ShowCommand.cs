// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Models;
using Idrak.Models.Abstractions;

namespace Idrak.Cli.Commands;

/// <summary>
/// <c>idrak show MODEL</c>: what a model is, read from its config, tokenizer files and weight headers (nothing is
/// loaded): family, parameters, layers, context, vocabulary, RoPE scaling, windows, chat template and tool-call format,
/// files and license.
/// </summary>
internal sealed class ShowCommand : Command
{
    public override string Name => "show";

    public override string Summary => "Family, parameters, layers, context, vocabulary, RoPE, windows, chat template, files and license of a model";

    public override string Usage => """
        MODEL [--template]

        Arguments:
          MODEL  a cached Hugging Face id, a folder, a .gguf file or an alias

        Options:
              --template  also print the chat template's source

        Reads the config, the tokenizer files and the weights' headers; the model is not loaded.

        Examples:
          idrak show Qwen/Qwen3-0.6B
          idrak show ./model.gguf --json
        """;

    public override IReadOnlyCollection<string> Flags => ["--template"];

    public override int Run(CommandContext context)
    {
        string name = context.Argument(0, "MODEL");
        var local = ModelCache.Locate(context, name);
        var info = ModelFacts.Read(local);

        var fields = new List<(string, string)>
        {
            ("Model", $"{name}{(name == (local.File ?? local.Folder) ? "" : $" ({local.File ?? local.Folder})")}"),
            ("Family", $"{info.Architecture}{(info.ModelType is null ? "" : $" ({info.ModelType})")}{(info.Loadable ? "" : $"  (not loadable: {info.Problem})")}"),
            ("Parameters", $"{Units.Count(info.Parameters)} ({info.Parameters:N0})"),
        };
        if (info.Spec is { } spec)
        {
            fields.Add(("Layers", $"{spec.Layers} · width {spec.Dim} · feed-forward {spec.FfDim}{(spec.Gated ? $" (gated, {spec.Activation})" : $" ({spec.Activation})")}"));
            if (spec.Experts > 0)
            {
                fields.Add(("Experts", $"{spec.Experts} of width {(spec.ExpertFfDim > 0 ? spec.ExpertFfDim : spec.FfDim)}, {spec.ExpertsPerToken} per token{(spec.SharedExpertFfDim > 0 ? $", a shared expert of width {spec.SharedExpertFfDim}" : "")} · {Units.Count(spec.ActiveParameterCount)} parameters active per token"));
            }

            fields.Add(("Attention", $"{spec.Heads} heads, {spec.KvHeads} key/value heads, head size {spec.HeadDim}{(spec.QkNorm ? ", query/key norms" : "")}"));
            fields.Add(("Context", $"{spec.MaxPositions:N0} positions"));
            fields.Add(("Vocabulary", $"{spec.Vocabulary:N0}{(spec.TieEmbeddings ? " (embeddings tied to the output head)" : "")}"));
            fields.Add(("RoPE", info.Rope));
            fields.Add(("Windows", info.Windows));
        }

        // A vision-language model's image side (vision_config), described without loading it: the keys most vision
        // configs share, as written (no family's defaults), and whether a vision family is registered for it (-P PLUGIN).
        JsonObject? vision = null;
        if (info.Config["vision_config"] is JsonObject visionConfig)
        {
            string? architecture = info.Config["architectures"] is JsonArray { Count: > 0 } names ? (string?)names[0] : null;
            bool registered = architecture is not null && VisionFamilies.Find(architecture) is not null;
            vision = new JsonObject
            {
                ["modelType"] = (string?)visionConfig["model_type"], ["family"] = architecture, ["registered"] = registered,
                ["layers"] = (int?)visionConfig["num_hidden_layers"], ["width"] = (int?)visionConfig["hidden_size"], ["heads"] = (int?)visionConfig["num_attention_heads"],
                ["imageSize"] = (int?)visionConfig["image_size"], ["patchSize"] = (int?)visionConfig["patch_size"],
            };
            string Part(string key, string label) => vision[key] is { } value ? $" · {label} {value}" : "";
            fields.Add(("Vision", $"{(string?)visionConfig["model_type"] ?? "vision encoder"}{Part("layers", "layers")}{Part("width", "width")}{Part("heads", "heads")}"
                + $"{Part("imageSize", "image px")}{Part("patchSize", "patch px")} · "
                + (registered ? $"vision family {architecture} registered" : $"vision family '{architecture}' not registered (load its plug-in with -P)")));
        }

        fields.Add(("Weights", $"{info.Format} · {Units.Bytes(info.WeightBytes)}"));
        fields.Add(("Chat", info.Template is null ? "no chat template" : $"chat template ({info.TemplateSource}), tool calls: {info.ToolCallFormat}"));
        fields.Add(("License", info.License ?? "not stated"));
        fields.Add(("Files", string.Join(", ", info.Files.Select(f => $"{f.Name} {Units.Bytes(f.Bytes)}"))));
        fields.AddRange(info.Notes.Select(note => ("Note", note)));
        context.Fields(fields);

        if (context.Flag("--template") && info.Template is not null)
        {
            context.Write("");
            context.Write(info.Template);
        }

        var json = new JsonObject
        {
            ["model"] = name,
            ["folder"] = local.Folder,
            ["file"] = local.File,
            ["architecture"] = info.Architecture,
            ["modelType"] = info.ModelType,
            ["loadable"] = info.Loadable,
            ["problem"] = info.Problem,
            ["parameters"] = info.Parameters,
            ["spec"] = info.Spec?.ToJson(),
            ["rope"] = info.Rope,
            ["windows"] = info.Windows,
            ["vision"] = vision,
            ["format"] = info.Format,
            ["weightBytes"] = info.WeightBytes,
            ["chatTemplate"] = info.Template is not null,
            ["toolCallFormat"] = info.ToolCallFormat,
            ["license"] = info.License,
            ["files"] = new JsonArray([.. info.Files.Select(f => (JsonNode)new JsonObject { ["name"] = f.Name, ["bytes"] = f.Bytes })]),
            ["notes"] = new JsonArray([.. info.Notes.Select(n => (JsonNode?)JsonValue.Create(n))]),
        };
        if (context.Flag("--template"))
        {
            json["template"] = info.Template;
        }

        context.WriteJson(json);
        return ExitCodes.Ok;
    }
}
