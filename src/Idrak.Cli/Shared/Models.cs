// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak.LanguageModels;

namespace Idrak.Cli.Shared;

/// <summary>
/// Model arguments, the same in every command: a Hugging Face id, a folder, a GGUF file, a .ikm package or an alias
/// from the config ("aliases": {"qwen": {"model": "Qwen/Qwen3-0.6B", "weights": "int8", "kv": "int8"}}), with the
/// weight format (<c>--weights</c>, <c>-w</c>), the KV cache format (<c>--kv</c>, <c>-k</c>), the context length and
/// an adapter folder.
/// </summary>
internal static class Models
{
    /// <summary>The options a command that loads a language model accepts.</summary>
    public static readonly string[] ValueOptions = ["--weights", "--kv", "--context", "--adapter"];

    /// <summary>Their short forms.</summary>
    public static readonly IReadOnlyDictionary<string, string> ShortForms = new Dictionary<string, string> { ["-w"] = "--weights", ["-k"] = "--kv" };

    /// <summary>What an alias or the options say: the model name and its settings.</summary>
    public sealed record ModelChoice(string Model, string? Weights, string? Kv, int? Context, string? Adapter);

    /// <summary>The model named by <paramref name="name"/> with the command's options applied over the alias's settings.</summary>
    public static ModelChoice Choose(CommandContext context, string name)
    {
        string model = name;
        string? weights = null, kv = null;
        if (context.Config.Object("aliases")?[name] is System.Text.Json.Nodes.JsonObject alias)
        {
            model = (string?)alias["model"] ?? name;
            weights = (string?)alias["weights"];
            kv = (string?)alias["kv"];
        }

        return new ModelChoice(model, context.Option("--weights") ?? weights, context.Option("--kv") ?? kv,
            context.Option("--context") is null ? null : context.IntOption("--context", 0), context.Option("--adapter"));
    }

    /// <summary>A local folder or file for the model: found in the cache or downloaded (with progress unless quiet).</summary>
    public static string Resolve(CommandContext context, string model) =>
        Directory.Exists(model) || File.Exists(model) ? model
        : LooksLocal(model) ? throw new UsageException($"Model not found: {model}. Give a Hugging Face id (owner/name), a model folder, a .gguf file or an alias from the config.")
        : ModelSource.Resolve(model, downloader: context.Quiet || context.Json ? null : new Idrak.Datasets.ConsoleStatus().CreateDownloader());

    // A path rather than a hub id: never looked up online.
    private static bool LooksLocal(string model) =>
        Path.IsPathRooted(model) || model.StartsWith('.') || model.StartsWith('~') || model.Contains('\\')
        || model.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase) || model.EndsWith(".ikm", StringComparison.OrdinalIgnoreCase);

    /// <summary>Loads a pretrained language model on the context's device with the chosen formats.</summary>
    public static PretrainedModel Load(CommandContext context, ModelChoice choice)
    {
        string folder = Resolve(context, choice.Model);
        string? weights = choice.Weights?.ToLowerInvariant();
        var watch = Stopwatch.StartNew();
        var model = PretrainedModel.Load(folder, new PretrainedOptions
        {
            Device = context.Device,
            Int8 = weights == "int8",
            Int4 = weights == "int4",
            BFloat16 = weights == "bf16" || weights == "bfloat16",
            PackedFormatName = weights is null or "int8" or "int4" or "bf16" or "bfloat16" or "float32" or "f32" ? null : choice.Weights,
            MaxPositions = choice.Context,
            MergeAdapter = choice.Adapter,
        });
        context.Detail($"loaded {choice.Model} in {watch.Elapsed.TotalSeconds:F1} s on {context.Device}");
        return model;
    }

    /// <summary>The KV cache format name for generation (null: the library's default).</summary>
    public static string? KvFormat(ModelChoice choice) => choice.Kv;

    /// <summary>The context window generation uses when <c>--context</c> is not given (the KV cache is sized by it).</summary>
    public const int DefaultContext = 4096;

    /// <summary>The context window for generating with <paramref name="model"/>: <c>--context</c> or <see cref="DefaultContext"/>, at most the model's.</summary>
    public static int ContextLength(ModelChoice choice, PretrainedModel model) => Math.Min(choice.Context ?? DefaultContext, model.MaxPositions);

    /// <summary>
    /// The KV cache layout named by <c>--kv</c>: float32 (f32), int8, bfloat16 (bf16) or any registered format; float32
    /// when not given.
    /// </summary>
    public static Idrak.Layers.KeyValueLayout CacheLayout(ModelChoice choice)
    {
        string name = choice.Kv?.ToLowerInvariant() switch { null or "f32" or "fp32" => "float32", "bf16" => "bfloat16", var other => other };
        try
        {
            return Idrak.Layers.KeyValueLayouts.Get(name);
        }
        catch (NotSupportedException)
        {
            throw new UsageException($"Unknown --kv format '{choice.Kv}'; use {string.Join(", ", Idrak.Layers.KeyValueLayouts.Names.Order(StringComparer.Ordinal))}, "
                                     + "or load a plug-in that registers it (--plugin).");
        }
    }

    /// <summary>A text generator for <paramref name="model"/> with the chosen KV cache format and context window.</summary>
    public static Idrak.Generation.TextGenerator CreateGenerator(PretrainedModel model, ModelChoice choice) =>
        model.CreateGenerator(CacheLayout(choice), ContextLength(choice, model));

    /// <summary>A chat generator for <paramref name="model"/> (its own chat template) with the chosen KV cache format and context window.</summary>
    public static Idrak.Generation.ChatGenerator CreateChat(PretrainedModel model, ModelChoice choice) =>
        model.ChatTemplate is null
            ? throw new InvalidOperationException($"{choice.Model} has no chat template (tokenizer_config.json); use 'idrak complete' for raw completion.")
            : model.CreateChat(CacheLayout(choice), ContextLength(choice, model));
}
