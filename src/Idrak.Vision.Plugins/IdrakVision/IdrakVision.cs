// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Imaging;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Vision;

namespace Idrak.Vision.Plugins.IdrakVision;

/// <summary>
/// IdrakVision's vision side (<see cref="IVisionFamily"/>): reads a checkpoint's vision part (<c>vision_tower.*</c>,
/// <c>multi_modal_projector.*</c>, config.json's <c>vision_config</c>, <c>projector_config</c>, image token ids and
/// <c>image_attention</c>) and makes a new one from a description (<see cref="Create"/>), every shape a setting.
/// </summary>
public sealed class IdrakVisionFamily : IVisionFamily
{
    /// <summary>The architecture name.</summary>
    public const string Architecture = "IdrakVisionForConditionalGeneration";

    /// <summary>The tower's tensors' prefix in a checkpoint.</summary>
    public const string TowerPrefix = "vision_tower.";

    /// <summary>The projector's tensors' prefix in a checkpoint.</summary>
    public const string ProjectorPrefix = "multi_modal_projector.";

    /// <inheritdoc />
    public string Name => Architecture;

    /// <inheritdoc />
    public IPretrainedVision? Read(VisionCheckpoint checkpoint)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        var config = checkpoint.Config;
        if (!checkpoint.Tensors.Contains(TowerPrefix + "embeddings.patch_embedding.weight"))
        {
            checkpoint.Notes.Add("The checkpoint has no vision tower: the model reads text only.");
            return null;
        }

        var vision = IdrakVisionConfig.FromJson(config["vision_config"] as JsonObject, config["projector_config"] as JsonObject);
        int textDim = PretrainedArchitectures.Get(checkpoint.Architecture).Spec(config, []).Dim;
        var tokens = new IdrakVisionTokens(
            (int?)config["image_token_index"] ?? throw new InvalidDataException("config.json has no image_token_index."),
            (int?)config["boi_token_index"], (int?)config["eoi_token_index"], vision.TokensPerImage);
        if ((int?)config["mm_tokens_per_image"] is int given && given != vision.TokensPerImage)
        {
            throw new InvalidDataException($"mm_tokens_per_image {given}, but the vision_config pools {vision.PatchesPerSide} x {vision.PatchesPerSide} patches into {vision.TokensPerImage}.");
        }

        var expected = VisionTransformerEncoder.Expected(vision.ToTransformer(), TowerPrefix).Concat(IdrakVisionProjector.Expected(vision, textDim, ProjectorPrefix)).ToList();
        foreach (var (name, shape) in expected)
        {
            if (!checkpoint.Tensors.Contains(name))
            {
                throw new InvalidDataException($"The checkpoint has no {name}.");
            }

            if (!checkpoint.Tensors.ShapeOf(name).SequenceEqual(shape))
            {
                throw new InvalidDataException($"{name}: shape {Tensor.FormatShape(checkpoint.Tensors.ShapeOf(name))}, expected {Tensor.FormatShape(shape)}.");
            }
        }

        string attention = (string?)config["image_attention"] ?? "image-blocks";
        return new IdrakVisionPart(vision, tokens, textDim, attention, [.. expected.Select(e => e.Name)], checkpoint.Open, checkpoint.Folder);
    }

    /// <summary>
    /// A new vision part from <paramref name="vision"/>: <c>{"vision_config": {...}, "projector_config": {...},
    /// "image_attention": "image-blocks" | "causal", "image_token": "&lt;image&gt;", "begin_image": "&lt;start_of_image&gt;",
    /// "end_image": "&lt;end_of_image&gt;", "init_std": 0.02, "preprocessing": {preprocessor_config.json}}</c>, every key
    /// optional. The tower's and projector's matrices, convolution and positions are drawn normal with <c>init_std</c>
    /// from <paramref name="seed"/>, norms start at one and biases at zero; the preprocessing defaults to the image size,
    /// bilinear, values scaled to [0, 1] and normalized with mean and deviation 0.5.
    /// </summary>
    public CreatedVision Create(JsonObject vision, int textDim, ITokenizer tokenizer, int seed)
    {
        ArgumentNullException.ThrowIfNull(vision);
        ArgumentNullException.ThrowIfNull(tokenizer);
        var config = IdrakVisionConfig.FromJson(vision["vision_config"] as JsonObject, vision["projector_config"] as JsonObject);
        int Token(string key, string fallback)
        {
            string text = (string?)vision[key] ?? fallback;
            var ids = tokenizer.Encode(text);
            return ids.Count == 1 ? ids[0] : throw new InvalidDataException($"The tokenizer has no single token '{text}' ({key}); add it as a special token (idrak data tokenizer --special).");
        }

        int image = Token("image_token", "<image>"), begin = Token("begin_image", "<start_of_image>"), end = Token("end_image", "<end_of_image>");
        string attention = (string?)vision["image_attention"] ?? "image-blocks";
        _ = IdrakVisionPart.AttentionRule(attention);
        float std = (float?)vision["init_std"] ?? 0.02f;
        var random = new Random(seed ^ 0x5eed);
        var tensors = new List<(string, int[], float[])>();
        foreach (var (name, shape) in VisionTransformerEncoder.Expected(config.ToTransformer(), TowerPrefix).Concat(IdrakVisionProjector.Expected(config, textDim, ProjectorPrefix)))
        {
            int count = shape.Aggregate(1, (a, b) => a * b);
            var values = new float[count];
            if (name.EndsWith(".bias", StringComparison.Ordinal))
            {
                // zeros
            }
            else if (name.Contains("layer_norm", StringComparison.Ordinal) || name.Contains("post_layernorm", StringComparison.Ordinal))
            {
                Array.Fill(values, 1f);
            }
            else
            {
                for (int i = 0; i < count; i += 2)
                {
                    // Box-Muller: two normal values from two uniform ones.
                    double u = 1.0 - random.NextDouble(), v = random.NextDouble();
                    double radius = Math.Sqrt(-2.0 * Math.Log(u));
                    values[i] = (float)(std * radius * Math.Cos(2 * Math.PI * v));
                    if (i + 1 < count)
                    {
                        values[i + 1] = (float)(std * radius * Math.Sin(2 * Math.PI * v));
                    }
                }
            }

            tensors.Add((name, shape, values));
        }

        var preprocessing = vision["preprocessing"] as JsonObject ?? new JsonObject
        {
            ["do_convert_rgb"] = true, ["do_resize"] = true, ["size"] = new JsonObject { ["height"] = config.ImageSize, ["width"] = config.ImageSize },
            ["resample"] = 2, ["do_rescale"] = true, ["rescale_factor"] = 1.0 / 255, ["do_normalize"] = true,
            ["image_mean"] = new JsonArray(0.5, 0.5, 0.5), ["image_std"] = new JsonArray(0.5, 0.5, 0.5),
        };
        var keys = new JsonObject
        {
            ["model_type"] = "idrak_vision",
            ["vision_config"] = config.VisionJson(),
            ["projector_config"] = config.ProjectorJson(),
            ["image_token_index"] = image,
            ["boi_token_index"] = begin,
            ["eoi_token_index"] = end,
            ["mm_tokens_per_image"] = config.TokensPerImage,
            ["image_attention"] = attention,
        };
        return new CreatedVision(keys, tensors, new Dictionary<string, string> { ["preprocessor_config.json"] = preprocessing.ToJsonString() });
    }
}

/// <summary>
/// An IdrakVision model's vision part: the prompt format (the template's <c>&lt;image&gt;</c> becomes begin, the image
/// tokens and end), the attention rule, and encoders built from the checkpoint; its tower and projector train in full
/// (<see cref="IVisionTuningPart"/>).
/// </summary>
public sealed class IdrakVisionPart : PretrainedVision, IVisionTuningPart
{
    private readonly Func<ITensorStore> _open;
    private readonly string? _folder;
    private readonly IReadOnlyCollection<string> _stored;
    private readonly string _attention;

    internal IdrakVisionPart(IdrakVisionConfig config, IdrakVisionTokens tokens, int textDim, string attention, IReadOnlyCollection<string> stored, Func<ITensorStore> open, string? folder)
    {
        Config = config;
        Tokens = tokens;
        Width = textDim;
        _attention = attention;
        _stored = stored;
        _open = open;
        _folder = folder;
        PromptFormat = new ImageTokenFormat("idrak-vision", tokens.ImageToken, tokens.ImageToken, tokens.BeginImage, tokens.EndImage);
    }

    /// <summary>The vision part's shape.</summary>
    public IdrakVisionConfig Config { get; }

    /// <summary>The image tokens.</summary>
    public IdrakVisionTokens Tokens { get; }

    /// <inheritdoc />
    public override string Family => IdrakVisionFamily.Architecture;

    /// <inheritdoc />
    public override int Width { get; }

    /// <inheritdoc />
    public override IImagePromptFormat PromptFormat { get; }

    /// <inheritdoc />
    public override IImageAttentionRule Attention => AttentionRule(_attention);

    /// <inheritdoc />
    public override IReadOnlyCollection<string> StoredTensors => _stored;

    // "image-blocks": an image's tokens see each other (the plug-in's rule); "causal": the library's.
    internal static IImageAttentionRule AttentionRule(string name) => name switch
    {
        "image-blocks" => ImageAttentionRules.Get(IdrakVisionPlugin.ImageBlocks),
        "causal" => ImageAttentionRules.Get(ImageAttentionRules.Causal),
        _ => throw new InvalidDataException($"image_attention '{name}' is not image-blocks or causal."),
    };

    /// <summary>The preprocessing: preprocessor_config.json in the model folder, else the image size, bilinear, mean and deviation 0.5.</summary>
    public ImagePreprocessor Preprocessor(bool grayscale = false)
    {
        var defaults = new ImagePreprocessor { Resize = true, Height = Config.ImageSize, Width = Config.ImageSize, Mean = [0.5f, 0.5f, 0.5f], Std = [0.5f, 0.5f, 0.5f], Grayscale = grayscale };
        return _folder is not null && File.Exists(Path.Combine(_folder, "preprocessor_config.json"))
            ? ImagePreprocessor.FromConfig(_folder, grayscale, defaults)
            : defaults;
    }

    /// <inheritdoc />
    public override IVisionEncoder CreateEncoder(VisionEncoderOptions? options = null)
    {
        var device = options?.Device ?? Device.Default;
        using var store = _open();
        var tower = VisionTransformerEncoder.FromTensors(Config.ToTransformer(), store, IdrakVisionFamily.TowerPrefix, device, EncoderWeights.Float32);
        try
        {
            var projector = IdrakVisionProjector.FromTensors(Config, Width, store, IdrakVisionFamily.ProjectorPrefix, device);
            return new IdrakVisionEncoder(this, tower, projector, Preprocessor(options?.Grayscale ?? false));
        }
        catch
        {
            tower.Dispose();
            throw;
        }
    }

    /// <summary>The tower and the projector train.</summary>
    public IReadOnlyList<string> TrainableParts { get; } = [VisionTuningParts.Projector, VisionTuningParts.Tower];

    /// <inheritdoc />
    public IReadOnlyList<Tensor> Parameters(IVisionEncoder encoder, string part)
    {
        var own = Own(encoder);
        return Part(part) switch
        {
            VisionTuningParts.Projector => [.. own.Projector.Parameters()],
            _ => [.. own.Tower.Parameters()],
        };
    }

    /// <inheritdoc />
    public Tensor Features(IVisionEncoder encoder, Tensor towerOutput) => Own(encoder).Projector.Forward(towerOutput);

    /// <inheritdoc />
    public Tensor Tower(IVisionEncoder encoder, Tensor pixelValues)
    {
        var own = Own(encoder);
        var input = pixelValues.Device == own.Device ? pixelValues : Tensor.From(pixelValues.ToArray(), pixelValues.Shape, own.Device);
        return own.Tower.ForwardTraining(input, ActivationMemory.CheckpointsBlocks);
    }

    /// <inheritdoc />
    public IReadOnlyDictionary<string, Tensor> Export(IVisionEncoder encoder, string part)
    {
        var own = Own(encoder);
        var values = Part(part) == VisionTuningParts.Projector ? own.Projector.Tensors(IdrakVisionFamily.ProjectorPrefix) : own.Tower.Tensors(IdrakVisionFamily.TowerPrefix);
        return values.ToDictionary(t => t.Name, t => Tensor.From(t.Values, t.Shape, Device.Cpu), StringComparer.Ordinal);
    }

    /// <inheritdoc />
    public IReadOnlyCollection<string> Import(IVisionEncoder encoder, IReadOnlyDictionary<string, Tensor> tensors)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        var own = Own(encoder);
        var values = tensors.ToDictionary(t => t.Key, t => t.Value.ToArray(), StringComparer.Ordinal);
        return [.. own.Tower.Load(IdrakVisionFamily.TowerPrefix, values), .. own.Projector.Load(IdrakVisionFamily.ProjectorPrefix, values)];
    }

    /// <inheritdoc />
    public IReadOnlyList<string> PartsOf(IEnumerable<string> names)
    {
        var parts = new List<string>();
        foreach (string name in names)
        {
            string? part = name.StartsWith(IdrakVisionFamily.ProjectorPrefix, StringComparison.Ordinal) ? VisionTuningParts.Projector
                : name.StartsWith(IdrakVisionFamily.TowerPrefix, StringComparison.Ordinal) ? VisionTuningParts.Tower : null;
            if (part is not null && !parts.Contains(part))
            {
                parts.Add(part);
            }
        }

        return parts;
    }

    /// <inheritdoc />
    public override string Describe() =>
        $"IdrakVision: {Config.ImageSize} x {Config.ImageSize}, {Config.Layers} layers of {Config.Dim}, {Tokens.TokensPerImage} tokens per image, image tokens attend: {Attention.Name}";

    private string Part(string part) => part?.Trim().ToLowerInvariant() switch
    {
        VisionTuningParts.Projector => VisionTuningParts.Projector,
        VisionTuningParts.Tower => VisionTuningParts.Tower,
        _ => throw new NotSupportedException($"The vision family {Family} does not offer '{part}' for training; it offers: {string.Join(", ", TrainableParts)}."),
    };

    private IdrakVisionEncoder Own(IVisionEncoder encoder)
    {
        ArgumentNullException.ThrowIfNull(encoder);
        return encoder is IdrakVisionEncoder own && ReferenceEquals(own.Vision, this) ? own
            : throw new ArgumentException($"{encoder.GetType().Name} is not an encoder of this IdrakVision part (make one with its CreateEncoder).", nameof(encoder));
    }
}
