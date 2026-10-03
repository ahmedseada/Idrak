// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.LanguageModels;

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
            fields.Add(("Attention", $"{spec.Heads} heads, {spec.KvHeads} key/value heads, head size {spec.HeadDim}{(spec.QkNorm ? ", query/key norms" : "")}"));
            fields.Add(("Context", $"{spec.MaxPositions:N0} positions"));
            fields.Add(("Vocabulary", $"{spec.Vocabulary:N0}{(spec.TieEmbeddings ? " (embeddings tied to the output head)" : "")}"));
            fields.Add(("RoPE", info.Rope));
            fields.Add(("Windows", info.Windows));
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
