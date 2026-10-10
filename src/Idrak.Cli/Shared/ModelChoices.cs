// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak.Models;
using Idrak.Nlp;

namespace Idrak.Cli.Shared;

/// <summary>
/// Model arguments, the same in every command: a Hugging Face id, a folder, a GGUF file, a .ikm package or an alias
/// from the config ("aliases": {"qwen": {"model": "Qwen/Qwen3-0.6B", "weights": "int8", "kv": "int8"}}), with the
/// weight format (<c>--weights</c>, <c>-w</c>), the KV cache format (<c>--kv</c>, <c>-k</c>), the context length and
/// an adapter folder; for a vision-language model, whether its images are read in grey (<c>--grayscale</c>, or
/// "grayscale": true in the alias), the image transforms run on its images first (<c>--image-transform
/// "grayscale,max_width=1024,contrast=1.5"</c>, or the alias's "image_transforms"; <c>--grayscale</c> is the
/// <c>grayscale</c> transform first) and its vision family's own options (<c>--vision-option KEY=VALUE</c>, repeatable,
/// over the alias's "vision_options": {"KEY": VALUE}).
/// </summary>
internal static class ModelChoices
{
    /// <summary>The options a command that loads a language model accepts.</summary>
    public static readonly string[] ValueOptions = ["--weights", "--kv", "--context", "--adapter"];

    /// <summary>The option giving a vision family's own options, KEY=VALUE (repeatable).</summary>
    public const string VisionOption = "--vision-option";

    /// <summary>Help lines for <see cref="VisionOption"/>, aligned as the commands' options are.</summary>
    public const string VisionOptionHelp = """
              --vision-option K=V an option of the model's vision family for its images (repeatable; the family names
                                 the keys it takes and refuses others; with the Gemma 3 plug-in, do_pan_and_scan=true
                                 adds crops of a tall or wide page); an alias can keep them ("vision_options")
        """;

    /// <summary>The option giving the image transforms, a pipeline such as grayscale,max_width=1024,contrast=1.5 (repeatable: joined in order).</summary>
    public const string ImageTransformOption = "--image-transform";

    /// <summary>Help lines for <see cref="ImageTransformOption"/>, aligned as the commands' options are.</summary>
    public const string ImageTransformHelp = """
              --image-transform P image transforms run on every image first, in order, as Pillow does them (P such as
                                 grayscale,max_width=1024,contrast=1.5; also max_height=N, resample=lanczos|bicubic|
                                 bilinear|box|hamming after a size, brightness=F, sharpness=F, autocontrast[=CUT],
                                 invert, jpeg=Q); "none" for none; an alias can keep them ("image_transforms")
        """;

    /// <summary>Their short forms.</summary>
    public static readonly IReadOnlyDictionary<string, string> ShortForms = new Dictionary<string, string> { ["-w"] = "--weights", ["-k"] = "--kv" };

    /// <summary>What an alias or the options say: the model name and its settings.</summary>
    public sealed record ModelChoice(string Model, string? Weights, string? Kv, int? Context, string? Adapter, bool Grayscale = false, VisionOptions? VisionOptions = null,
        ImageTransformPipeline? ImageTransforms = null)
    {
        /// <summary>
        /// The preparation saved with the adapter (<see cref="TuningImages.FileName"/>: how its images were prepared in
        /// tuning, and for which vision family), or null. Its transforms, grayscale and vision options are in this choice
        /// unless the command line gave its own.
        /// </summary>
        public TuningImages? AdapterImages { get; init; }

        /// <summary>The transforms every image goes through: <c>grayscale</c> first when asked (and not already a step), then <see cref="ImageTransforms"/>.</summary>
        public ImageTransformPipeline Transforms => Grayscale && ImageTransforms?.Contains("grayscale") != true
            ? ImageTransformPipeline.Parse("grayscale").Then(ImageTransforms)
            : ImageTransforms ?? ImageTransformPipeline.Empty;
    }

    /// <summary>The model named by <paramref name="name"/> with the command's options applied over the alias's settings.</summary>
    public static ModelChoice Choose(CommandContext context, string name)
    {
        string model = name;
        string? weights = null, kv = null;
        bool grayscale = false;
        var vision = VisionOptions.Empty;
        ImageTransformPipeline? transforms = null;
        if (context.Config.Object("aliases")?[name] is System.Text.Json.Nodes.JsonObject alias)
        {
            model = (string?)alias["model"] ?? name;
            weights = (string?)alias["weights"];
            kv = (string?)alias["kv"];
            grayscale = alias["grayscale"] is System.Text.Json.Nodes.JsonValue grey && grey.TryGetValue(out bool value) && value;
            if (alias["vision_options"] is System.Text.Json.Nodes.JsonObject saved)
            {
                vision = Parsed(() => VisionOptions.FromJson(saved), $"the alias {name}'s \"vision_options\"");
            }

            if (alias["image_transforms"] is { } pipeline)
            {
                transforms = Parsed(() => ImageTransformPipeline.FromJson(pipeline), $"the alias {name}'s \"image_transforms\"");
            }
        }

        // An adapter tuned on images: its preparation (transforms, grayscale, vision options) over the alias's; the command
        // line's over both (--image-transform replaces its transforms and grayscale, --vision-option its options key by key).
        string? adapter = context.Option("--adapter");
        var tuned = AdapterImages(adapter);
        var given = ImageTransformsOf(context);
        if (tuned is not null)
        {
            vision = vision.With(tuned.VisionOptions);
            (transforms, grayscale) = given is null ? (tuned.Transforms, tuned.Grayscale) : (transforms, false);
        }

        vision = vision.With(VisionOptionsOf(context));
        return new ModelChoice(model, context.Option("--weights") ?? weights, context.Option("--kv") ?? kv,
            context.Option("--context") is null ? null : context.IntOption("--context", 0), adapter,
            grayscale || context.Flag("--grayscale"), vision.Count > 0 ? vision : null, given ?? transforms) { AdapterImages = tuned };
    }

    /// <summary>The image preparation an adapter folder was tuned with (<see cref="TuningImages.FileName"/>), or null when it has none.</summary>
    public static TuningImages? AdapterImages(string? adapter)
    {
        if (adapter is null || !Directory.Exists(adapter))
        {
            return null;
        }

        try
        {
            return TuningImages.Read(adapter);
        }
        catch (Exception ex) when (ex is InvalidDataException or IOException)
        {
            throw new UsageException($"--adapter {adapter}: {ex.Message}");
        }
    }

    /// <summary>
    /// Throws unless the adapter's saved preparation (<see cref="ModelChoice.AdapterImages"/>) was tuned on
    /// <paramref name="family"/> (a model's vision family; null for a text-only model).
    /// </summary>
    public static void CheckAdapterFamily(ModelChoice choice, string? family)
    {
        try
        {
            choice.AdapterImages?.ThrowIfOtherFamily(family, $"The adapter {choice.Adapter}");
        }
        catch (InvalidOperationException ex)
        {
            throw new UsageException($"{ex.Message} (the model given: {choice.Model})");
        }
    }

    /// <summary>The command's <c>--image-transform</c> pipeline (several joined in order), or null when not given.</summary>
    public static ImageTransformPipeline? ImageTransformsOf(CommandContext context)
    {
        var given = context.Options(ImageTransformOption);
        return given.Count == 0 ? null : Parsed(() => ImageTransformPipeline.Parse(string.Join(",", given)), ImageTransformOption);
    }

    /// <summary>The command's <c>--vision-option KEY=VALUE</c> options (empty when none).</summary>
    public static VisionOptions VisionOptionsOf(CommandContext context) =>
        Parsed(() => VisionOptions.Parse(context.Options(VisionOption)), VisionOption);

    private static T Parsed<T>(Func<T> parse, string where)
    {
        try
        {
            return parse();
        }
        catch (Exception ex) when (ex is FormatException or ArgumentException or System.Text.Json.JsonException)
        {
            throw new UsageException($"{where}: {ex.Message}");
        }
    }

    /// <summary>
    /// A local folder for the model, the same way in every command: a folder as it is; a .gguf file prepared under the
    /// command's cache folder (<c>--cache</c>); a Hugging Face id or a pulled GGUF file from the cache first (no network,
    /// its last use recorded for <c>idrak list</c>), else downloaded into the cache folder with a progress line. With
    /// <c>--offline</c> a model the cache lacks is an error naming <c>idrak pull</c>.
    /// </summary>
    public static string Resolve(CommandContext context, string model)
    {
        if (Directory.Exists(model))
        {
            return model;
        }

        if (File.Exists(model))
        {
            return model.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase) ? GgufModel.Prepare(model, context.CacheFolder) : model;
        }

        if (LooksLocal(model))
        {
            throw new UsageException($"Model not found: {model}. Give a Hugging Face id (owner/name), a model folder, a .gguf file or an alias from the config.");
        }

        if (ModelCache.TryLocate(context, model) is { } local)
        {
            context.Detail($"{model}: {local.File ?? local.Folder} (cached)");
            return local.Folder;
        }

        if (context.Offline)
        {
            throw new InvalidOperationException($"{model} is not in the cache ({context.CacheFolder}) and --offline allows no download; run 'idrak pull {model}' while online.");
        }

        if (ModelSources.For(model) is null || ModelSource.IsModelId(model) && model.Contains(':'))
        {
            // A GGUF file of a repository (owner/name/FILE.gguf, owner/name:TAG) is fetched by pull, not by loading.
            throw new InvalidOperationException($"{model} is not in the cache ({context.CacheFolder}). Download it first: idrak pull {model}");
        }

        string folder = ModelSource.Resolve(model, token: Hub.Token(context), downloader: Http.Downloader(context));
        ModelCache.Touch(context.CacheFolder, folder);
        return folder;
    }

    // A path rather than a hub id (or a pulled GGUF file's owner/name/FILE.gguf): never looked up online.
    private static bool LooksLocal(string model) =>
        Path.IsPathRooted(model) || model.StartsWith('.') || model.StartsWith('~') || model.Contains('\\')
        || model.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase) && model.Count(c => c == '/') != 2
        || model.EndsWith(".ikm", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// Loads a pretrained language model with the chosen formats on <paramref name="device"/> (default: the command's
    /// device), with a progress line while it loads.
    /// </summary>
    public static PretrainedModel Load(CommandContext context, ModelChoice choice, Device? device = null)
    {
        device ??= context.Device;
        string folder = Resolve(context, choice.Model);
        string? weights = choice.Weights?.ToLowerInvariant();
        if (choice.AdapterImages?.Family is not null && Path.Combine(folder, "config.json") is var configFile && File.Exists(configFile)
            && System.Text.Json.Nodes.JsonNode.Parse(File.ReadAllText(configFile)) is System.Text.Json.Nodes.JsonObject config)
        {
            // Before reading any weight: a vision family is registered under its architecture's name (VisionFamilies).
            CheckAdapterFamily(choice, Idrak.Models.Abstractions.VisionFamilies.HasVision(config) ? (string?)config["architectures"]?[0] : null);
        }

        var watch = Stopwatch.StartNew();
        PretrainedModel model;
        using (var progress = new ProgressLine(context, $"loading {choice.Model} on {device}", unit: ProgressUnit.Elapsed))
        {
            model = PretrainedModel.Load(folder, new PretrainedOptions
            {
                Device = device,
                Int8 = weights == "int8",
                Int4 = weights == "int4",
                BFloat16 = weights is "bf16" or "bfloat16",
                PackedFormatName = weights is null or "int8" or "int4" or "bf16" or "bfloat16" or "float32" or "f32" ? null : choice.Weights,
                MaxPositions = choice.Context,
                MergeAdapter = choice.Adapter,
            });
            progress.Clear();
        }

        context.Detail($"loaded {choice.Model} in {watch.Elapsed.TotalSeconds:F1} s on {device}");
        try
        {
            CheckAdapterFamily(choice, model.Vision?.Family);
        }
        catch
        {
            model.Dispose();
            throw;
        }

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
    public static Idrak.Abstraction.Generation.KeyValueLayout CacheLayout(ModelChoice choice)
    {
        string name = choice.Kv?.ToLowerInvariant() switch { null or "f32" or "fp32" => "float32", "bf16" => "bfloat16", var other => other };
        try
        {
            return Idrak.Abstraction.Generation.KeyValueLayouts.Get(name);
        }
        catch (NotSupportedException)
        {
            throw new UsageException($"Unknown --kv format '{choice.Kv}'; use {string.Join(", ", Idrak.Abstraction.Generation.KeyValueLayouts.Names.Order(StringComparer.Ordinal))}, "
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
