// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak.Models;
using Idrak.Text;

namespace Idrak.Cli.Shared;

/// <summary>
/// Model arguments, the same in every command: a Hugging Face id, a folder, a GGUF file, a .ikm package or an alias
/// from the config ("aliases": {"qwen": {"model": "Qwen/Qwen3-0.6B", "kv": "float32"}}), with the KV cache format
/// (<c>--kv</c>, <c>-k</c>), the context length and
/// an adapter folder; for a vision-language model, whether its images are read in grey (<c>--grayscale</c>, or
/// "grayscale": true in the alias), the image transforms run on its images first (<c>--image-transform
/// "grayscale,max_width=1024,contrast=1.5"</c>, or the alias's "image_transforms"; <c>--grayscale</c> is the
/// <c>grayscale</c> transform first) and its vision family's own options (<c>--vision-option KEY=VALUE</c>, repeatable,
/// over the alias's "vision_options": {"KEY": VALUE}).
/// </summary>
internal static class ModelChoices
{
    /// <summary>The options a command that loads a language model accepts.</summary>
    public static readonly string[] ValueOptions = ["--kv", "--context", "--adapter"];

    /// <summary>
    /// The option giving the decoder layers kept on the device when the model is split with the CPU (a count, or auto: as the
    /// device's measured free memory allows); over IDRAK_DEVICE_LAYERS (<see cref="Idrak.Models.Abstractions.WeightPlacements"/>).
    /// </summary>
    public const string DeviceLayersOption = "--device-layers";

    /// <summary>Help lines for <see cref="DeviceLayersOption"/>, aligned as the commands' options are.</summary>
    public const string DeviceLayersHelp = """
              --device-layers N  decoder layers kept on the GPU when the model does not fit it; the rest are computed by
                                 the CPU from system memory (auto: as the GPU's free memory allows; IDRAK_DEVICE_LAYERS;
                                 IDRAK_WEIGHT_PLACEMENT device, split, stream or auto chooses how)
              --moe-experts P    where a mixture-of-experts model's experts live: resident (all on the GPU), offload
                                 (the most used on the GPU, the rest in system memory, copied in when picked) or auto
                                 (the default: resident when it fits; IDRAK_MOE_EXPERTS, IDRAK_MOE_DEVICE_EXPERTS_MB,
                                 IDRAK_MOE_COLD=cpu computes cold experts on the CPU)
        """;

    /// <summary>
    /// The option naming where a mixture-of-experts model's experts live (resident, offload, auto or an expert
    /// placement, plan 37's R1); over IDRAK_MOE_EXPERTS (<see cref="Idrak.Models.Abstractions.ExpertPlacements"/>).
    /// </summary>
    public const string MoeExpertsOption = "--moe-experts";

    /// <summary>The option giving the type every projection's weight is written as (a built-in type's name, matched where the text comes in).</summary>
    public const string WeightsOption = "--weights";

    /// <summary>Help lines for <see cref="WeightsOption"/>, aligned as the commands' options are.</summary>
    public const string WeightsHelp = """
          -w, --weights TYPE     every projection's weight and the token table written as TYPE by its type's encoder
                                 (a built-in type's name: q4_k, q8_0, int8, bfloat16, float8_e4m3, ...; int8 and int16
                                 rows carry a float32 scale a row; the alias's "weights"); kept as its checkpoint
                                 stores it when not given
        """;

    /// <summary>The option giving a vision family's own options, KEY=VALUE (repeatable).</summary>
    public const string VisionOption = "--vision-option";

    /// <summary>Help lines for <see cref="VisionOption"/>, aligned as the commands' options are.</summary>
    public const string VisionOptionHelp = """
              --vision-option K=V an option of the model's vision family for its images (repeatable; the family names
                                 the keys it takes and refuses others; Gemma 3's do_pan_and_scan=true adds crops of a
                                 tall or wide page); an alias can keep them ("vision_options")
        """;

    /// <summary>The option giving the image transforms, a pipeline such as grayscale,max_width=1024,contrast=1.5 (repeatable: joined in order).</summary>
    public const string ImageTransformOption = "--image-transform";

    /// <summary>Help lines for <see cref="ImageTransformOption"/>, aligned as the commands' options are.</summary>
    public const string ImageTransformHelp = """
              --image-transform P image transforms run on every image first, in order, as Pillow does them (P such as
                                 grayscale,max_width=1024,contrast=1.5); "none" for none: picking them by their names
                                 is not on the new design yet
        """;

    /// <summary>Their short forms.</summary>
    public static readonly IReadOnlyDictionary<string, string> ShortForms = new Dictionary<string, string> { ["-k"] = "--kv" };

    /// <summary>What an alias or the options say: the model name and its settings.</summary>
    public sealed record ModelChoice(string Model, string? Kv, int? Context, string? Adapter, bool Grayscale = false, VisionOptions? VisionOptions = null)
    {
        /// <summary>
        /// The preparation saved with the adapter (<see cref="TuningImages.FileName"/>: how its images were prepared in
        /// tuning, and for which vision family), or null. Its grayscale and vision options are in this choice unless the
        /// command line gave its own.
        /// </summary>
        public TuningImages? AdapterImages { get; init; }

        /// <summary>The decoder layers kept on the device (<c>--device-layers</c>, or the alias's "device_layers"), or null to size them (auto).</summary>
        public int? DeviceLayers { get; init; }

        /// <summary>Where a mixture-of-experts model's experts live (<c>--moe-experts</c>, or the alias's "moe_experts"), or null for IDRAK_MOE_EXPERTS, else auto.</summary>
        public string? MoeExperts { get; init; }

        /// <summary>The type every projection's weight is written as (<c>--weights</c>, or the alias's "weights"), or null: kept as stored.</summary>
        public IElementType? Weights { get; init; }
    }

    /// <summary>The model named by <paramref name="name"/> with the command's options applied over the alias's settings.</summary>
    public static ModelChoice Choose(CommandContext context, string name)
    {
        string model = name;
        string? kv = null, deviceLayers = null, moeExperts = null, weights = null;
        bool grayscale = false;
        var vision = VisionOptions.Empty;
        if (context.Config.Object("aliases")?[name] is System.Text.Json.Nodes.JsonObject alias)
        {
            model = (string?)alias["model"] ?? name;
            kv = (string?)alias["kv"];
            deviceLayers = alias["device_layers"]?.ToString();
            moeExperts = (string?)alias["moe_experts"];
            weights = (string?)alias["weights"];
            grayscale = alias["grayscale"] is System.Text.Json.Nodes.JsonValue grey && grey.TryGetValue(out bool value) && value;
            if (alias["vision_options"] is System.Text.Json.Nodes.JsonObject saved)
            {
                vision = Parsed(() => VisionOptions.FromJson(saved), $"the alias {name}'s \"vision_options\"");
            }

            NoImageTransforms(alias["image_transforms"]?.ToString(), $"the alias {name}'s \"image_transforms\"");
        }

        // An adapter tuned on images: its preparation (grayscale, vision options) over the alias's; the command line's
        // over both (--vision-option its options key by key).
        NoImageTransforms(context.Options(ImageTransformOption) is { Count: > 0 } given ? string.Join(",", given) : null, ImageTransformOption);
        string? adapter = context.Option("--adapter");
        var tuned = AdapterImages(adapter);
        if (tuned is not null)
        {
            vision = vision.With(tuned.VisionOptions);
            grayscale = tuned.Grayscale;
        }

        vision = vision.With(VisionOptionsOf(context));
        return new ModelChoice(model, context.Option("--kv") ?? kv,
            context.Option("--context") is null ? null : context.IntOption("--context", 0), adapter,
            grayscale || context.Flag("--grayscale"), vision.Count > 0 ? vision : null)
        {
            AdapterImages = tuned,
            DeviceLayers = Parsed(() => Idrak.Models.Abstractions.WeightPlacements.ParseDeviceLayers(context.Option(DeviceLayersOption) ?? deviceLayers, DeviceLayersOption), DeviceLayersOption),
            MoeExperts = MoeExpertsOf(context.Option(MoeExpertsOption) ?? moeExperts),
            Weights = WeightsOf(context.Option(WeightsOption) ?? weights),
        };
    }

    // The built-in type a person's text names: the list walked, each type's own name compared (no table of names).
    private static IElementType? WeightsOf(string? name)
    {
        if (name is null)
        {
            return null;
        }

        foreach (var type in Idrak.BuiltInTypes.All.Types)
        {
            if (string.Equals(type.Name, name.Trim(), StringComparison.OrdinalIgnoreCase))
            {
                return type;
            }
        }

        throw new UsageException($"{WeightsOption} {name}: no built-in type has that name ({string.Join(", ", Idrak.BuiltInTypes.All.Types.Select(t => t.Name))}).");
    }

    // An expert placement's name, or null when none is given: picking it by name is not on the new design yet.
    private static string? MoeExpertsOf(string? name) => name is null ? null
        : throw new UsageException($"{MoeExpertsOption} {name}: picking an expert placement by its name is not on the new design yet.");

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

    /// <summary>
    /// Throws when image transforms are given (<paramref name="text"/>: not empty, "none" or an empty list): picking
    /// the image transforms by their names is not on the new design yet.
    /// </summary>
    /// <exception cref="UsageException">Image transforms are given.</exception>
    public static void NoImageTransforms(string? text, string where)
    {
        if (!string.IsNullOrWhiteSpace(text) && text.Trim() is not "[]" && !text.Trim().Equals("none", StringComparison.OrdinalIgnoreCase))
        {
            throw new UsageException($"{where}: picking the image transforms '{text}' by their names is not on the new design yet.");
        }
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
            return model.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase) ? GgufModel.Prepare(model, context.CacheFolder, Idrak.Text.Plugins.BuiltInFamilies.All) : model;
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

        if (ModelSource.IsModelId(model) && model.Contains(':'))
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
    /// Loads a pretrained language model on <paramref name="device"/> (default: the command's device), with a progress line
    /// while it loads.
    /// </summary>
    public static PretrainedModel Load(CommandContext context, ModelChoice choice, Device? device = null)
    {
        device ??= context.Device;
        string folder = Resolve(context, choice.Model);
        if (choice.AdapterImages?.Family is not null && Path.Combine(folder, "config.json") is var configFile && File.Exists(configFile)
            && System.Text.Json.Nodes.JsonNode.Parse(File.ReadAllText(configFile)) is System.Text.Json.Nodes.JsonObject config)
        {
            // Before reading any weight: a vision family is the one of its architecture's name.
            CheckAdapterFamily(choice, config["vision_config"] is System.Text.Json.Nodes.JsonObject ? (string?)config["architectures"]?[0] : null);
        }

        var watch = Stopwatch.StartNew();
        PretrainedModel model;
        using (var progress = new ProgressLine(context, $"loading {choice.Model} on {device}", unit: ProgressUnit.Elapsed))
        {
            model = PretrainedModel.Load(folder, new PretrainedOptions
            {
                Device = device,
                MaxPositions = choice.Context,
                MergeAdapter = choice.Adapter,
                AutoWeightsContext = choice.Context ?? DefaultContext,
                DeviceLayers = choice.DeviceLayers,
                ExpertPlacement = choice.MoeExperts,
                Weights = choice.Weights,
                Families = Idrak.Text.Plugins.BuiltInFamilies.All,
                ChatTemplateReader = Idrak.Text.JinjaChatTemplate.Reader,
            });
            progress.Clear();
        }

        if (model.Placement is { } placed)
        {
            // A model split with the CPU or streamed from system memory is said on the error output (it is slower for it).
            if (placed.Split || placed.Stream)
            {
                context.Error($"placement {placed}");
            }
            else
            {
                context.Detail($"placement {placed}");
            }
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

    /// <summary>The KV cache layout of <c>--kv</c> (or the alias's "kv"): float32 (<see cref="CacheLayout(string?, string)"/>).</summary>
    public static Idrak.Abstraction.Generation.IKeyValueLayout CacheLayout(ModelChoice choice)
    {
        ArgumentNullException.ThrowIfNull(choice);
        return CacheLayout(choice.Kv, "--kv");
    }

    /// <summary>
    /// The built-in float32 layout, when <paramref name="name"/> is not given, auto or float32; any other name is refused,
    /// naming <paramref name="option"/>: picking a format by its name is not on the new design yet.
    /// </summary>
    public static Idrak.Abstraction.Generation.IKeyValueLayout CacheLayout(string? name, string option) =>
        string.IsNullOrWhiteSpace(name) || name.Trim().Equals("auto", StringComparison.OrdinalIgnoreCase) || name.Trim().Equals("float32", StringComparison.OrdinalIgnoreCase)
            ? Idrak.Abstraction.Generation.KeyValueLayouts.For(KeyValueFormat.Float32)
            : throw new UsageException($"{option} {name}: the key/value cache is float32 only; picking another format by its name is not on the new design yet.");

    /// <summary>A text generator for <paramref name="model"/> with the chosen KV cache format and context window.</summary>
    public static Idrak.Generation.TextGenerator CreateGenerator(PretrainedModel model, ModelChoice choice) =>
        model.CreateGenerator(CacheLayout(choice), ContextLength(choice, model));

    /// <summary>A chat generator for <paramref name="model"/> (its own chat template) with the chosen KV cache format and context window.</summary>
    public static Idrak.Text.Generation.ChatGenerator CreateChat(PretrainedModel model, ModelChoice choice) =>
        model.ChatTemplate is null
            ? throw new InvalidOperationException($"{choice.Model} has no chat template (tokenizer_config.json); use 'idrak complete' for raw completion.")
            : model.CreateChat(CacheLayout(choice), ContextLength(choice, model));
}
