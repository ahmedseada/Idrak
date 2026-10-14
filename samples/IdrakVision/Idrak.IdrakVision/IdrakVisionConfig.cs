// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;

namespace Idrak.IdrakVision;

/// <summary>
/// An IdrakVision vision part's shape, every value a setting (config.json's <c>vision_config</c> and
/// <c>projector_config</c>): the image and patch sizes, the tower's width, layers, heads and feed-forward size, its
/// activation and norm epsilon, the pooling of the patch grid, and the projector's depth, activation and biases.
/// </summary>
public sealed record IdrakVisionConfig
{
    /// <summary>The tower's width.</summary>
    public int Dim { get; init; } = 768;

    /// <summary>The tower's feed-forward size.</summary>
    public int FfDim { get; init; } = 3072;

    /// <summary>Encoder layers.</summary>
    public int Layers { get; init; } = 12;

    /// <summary>Attention heads.</summary>
    public int Heads { get; init; } = 12;

    /// <summary>The square image the preprocessing gives (pixels a side).</summary>
    public int ImageSize { get; init; } = 448;

    /// <summary>The square patch (pixels a side).</summary>
    public int PatchSize { get; init; } = 14;

    /// <summary>Image channels (3 for RGB).</summary>
    public int Channels { get; init; } = 3;

    /// <summary>The LayerNorms' epsilon.</summary>
    public float LayerNormEpsilon { get; init; } = 1e-6f;

    /// <summary>The encoder MLP's activation: gelu_pytorch_tanh, gelu, quick_gelu or relu.</summary>
    public string Activation { get; init; } = "gelu_pytorch_tanh";

    /// <summary>The patch grid is average-pooled PoolSize x PoolSize into one image token.</summary>
    public int PoolSize { get; init; } = 2;

    /// <summary>The projector's linear layers (1: a single projection).</summary>
    public int ProjectorDepth { get; init; } = 2;

    /// <summary>The activation between the projector's layers.</summary>
    public string ProjectorActivation { get; init; } = "gelu_pytorch_tanh";

    /// <summary>Biases on the projector's layers.</summary>
    public bool ProjectorBias { get; init; } = true;

    /// <summary>Patches a side.</summary>
    public int PatchesPerSide => ImageSize / PatchSize;

    /// <summary>Patches of an image.</summary>
    public int Patches => PatchesPerSide * PatchesPerSide;

    /// <summary>Image tokens of an image (the pooled grid).</summary>
    public int TokensPerImage => (PatchesPerSide / PoolSize) * (PatchesPerSide / PoolSize);

    /// <summary>Checks the sizes fit together.</summary>
    /// <exception cref="InvalidDataException">They do not.</exception>
    public IdrakVisionConfig Checked()
    {
        if (Dim < 1 || FfDim < 1 || Layers < 1 || Heads < 1 || ImageSize < 1 || PatchSize < 1 || Channels < 1 || PoolSize < 1 || ProjectorDepth < 1)
        {
            throw new InvalidDataException("vision_config: every size must be positive.");
        }

        if (Dim % Heads != 0)
        {
            throw new InvalidDataException($"vision_config: hidden_size {Dim} is not a multiple of num_attention_heads {Heads}.");
        }

        if (ImageSize % PatchSize != 0 || PatchesPerSide % PoolSize != 0)
        {
            throw new InvalidDataException($"vision_config: image_size {ImageSize} is not a whole number of patches of {PatchSize}, pooled {PoolSize} x {PoolSize}.");
        }

        _ = Layers2.Activation(Activation);
        _ = Layers2.Activation(ProjectorActivation);
        return this;
    }

    /// <summary>The config from <c>vision_config</c> and <c>projector_config</c> (keys left out keep their defaults).</summary>
    public static IdrakVisionConfig FromJson(JsonObject? vision, JsonObject? projector)
    {
        var d = new IdrakVisionConfig();
        return new IdrakVisionConfig
        {
            Dim = (int?)vision?["hidden_size"] ?? d.Dim,
            FfDim = (int?)vision?["intermediate_size"] ?? d.FfDim,
            Layers = (int?)vision?["num_hidden_layers"] ?? d.Layers,
            Heads = (int?)vision?["num_attention_heads"] ?? d.Heads,
            ImageSize = (int?)vision?["image_size"] ?? d.ImageSize,
            PatchSize = (int?)vision?["patch_size"] ?? d.PatchSize,
            Channels = (int?)vision?["num_channels"] ?? d.Channels,
            LayerNormEpsilon = (float?)vision?["layer_norm_eps"] ?? d.LayerNormEpsilon,
            Activation = (string?)vision?["hidden_act"] ?? d.Activation,
            PoolSize = (int?)vision?["pool_size"] ?? d.PoolSize,
            ProjectorDepth = (int?)projector?["depth"] ?? d.ProjectorDepth,
            ProjectorActivation = (string?)projector?["hidden_act"] ?? d.ProjectorActivation,
            ProjectorBias = (bool?)projector?["bias"] ?? d.ProjectorBias,
        }.Checked();
    }

    /// <summary>config.json's <c>vision_config</c>.</summary>
    public JsonObject VisionJson() => new()
    {
        ["hidden_size"] = Dim, ["intermediate_size"] = FfDim, ["num_hidden_layers"] = Layers, ["num_attention_heads"] = Heads,
        ["num_channels"] = Channels, ["image_size"] = ImageSize, ["patch_size"] = PatchSize,
        ["layer_norm_eps"] = JsonValue.Create(double.Parse(LayerNormEpsilon.ToString("R", CultureInfo.InvariantCulture), CultureInfo.InvariantCulture)),
        ["hidden_act"] = Activation, ["pool_size"] = PoolSize,
    };

    /// <summary>config.json's <c>projector_config</c>.</summary>
    public JsonObject ProjectorJson() => new() { ["hidden_act"] = ProjectorActivation, ["depth"] = ProjectorDepth, ["bias"] = ProjectorBias };
}

/// <summary>The image tokens' ids: the one repeated per image token (also the marker the template writes), and the ones around an image.</summary>
/// <param name="ImageToken">The soft token (and the template's marker).</param>
/// <param name="BeginImage">Before an image's tokens, or null.</param>
/// <param name="EndImage">After them, or null.</param>
/// <param name="TokensPerImage">Soft tokens of an image.</param>
public sealed record IdrakVisionTokens(int ImageToken, int? BeginImage, int? EndImage, int TokensPerImage);

// The activations the tower and projector take, by their configuration names.
internal static class Layers2
{
    public static Func<Module> Activation(string name) => name switch
    {
        "gelu_pytorch_tanh" or "gelu_new" or "gelu_fast" => () => new Idrak.Layers.GELU(),
        "gelu" => () => new Idrak.Layers.ExactGELU(),
        "quick_gelu" => () => new Idrak.Layers.QuickGELU(),
        "relu" => () => new Idrak.Layers.ReLU(),
        _ => throw new InvalidDataException($"Activation '{name}' is not one IdrakVision takes (gelu_pytorch_tanh, gelu, quick_gelu, relu)."),
    };
}
