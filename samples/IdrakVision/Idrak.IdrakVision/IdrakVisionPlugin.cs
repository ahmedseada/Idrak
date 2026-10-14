// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Models;
using Idrak.Models.Abstractions;

namespace Idrak.IdrakVision;

/// <summary>
/// Registers IdrakVision: its text decoder (a Llama decoder read from <c>text_config</c>, its tensors under
/// <c>language_model.</c>, written back from a spec) in <see cref="PretrainedArchitectures"/>, its vision part in
/// <see cref="VisionFamilies"/>, and the image attention rule "idrak-image-blocks" (an image's tokens see each other, the
/// rest causal) in <see cref="ImageAttentionRules"/>, all under <see cref="IdrakVisionFamily.Architecture"/>.
/// </summary>
public static class IdrakVisionPlugin
{
    /// <summary>The image attention rule's name: an image's tokens attend to each other in both directions.</summary>
    public const string ImageBlocks = "idrak-image-blocks";

    /// <summary>Registers the family (its text decoder, vision part and attention rule).</summary>
    public static void Register()
    {
        ImageAttentionRules.Register(new ImageBlockRule());
        PretrainedArchitectures.Register(IdrakVisionFamily.Architecture, TextDecoder());
        VisionFamilies.Register(new IdrakVisionFamily());
    }

    /// <summary>The entry the command line calls for <c>-P Idrak.IdrakVision.dll</c>.</summary>
    public static void RegisterIdrakPlugin() => Register();

    /// <summary>Removes the registrations <see cref="Register"/> made.</summary>
    public static void Unregister()
    {
        VisionFamilies.Unregister(IdrakVisionFamily.Architecture);
        PretrainedArchitectures.Unregister(IdrakVisionFamily.Architecture);
        ImageAttentionRules.Unregister(ImageBlocks);
    }

    // The text side: Llama's spec from text_config, its tensors under language_model., a spec written as text_config.
    private static PretrainedArchitecture TextDecoder()
    {
        var llama = PretrainedFamilies.LlamaStyle((_, spec, _) => spec, "llama");
        return new PretrainedArchitecture
        {
            Spec = (config, notes) => llama.Spec(config["text_config"] as JsonObject
                ?? throw new InvalidDataException($"An {IdrakVisionFamily.Architecture} config.json has a \"text_config\"."), notes),
            TensorName = name => PretrainedFamilies.LlamaTensorName(name) is { } stored ? "language_model." + stored : null,
            Config = spec => new JsonObject { ["text_config"] = llama.Config!(spec), ["tie_word_embeddings"] = spec.TieEmbeddings },
        };
    }

    private sealed class ImageBlockRule : IImageAttentionRule
    {
        public string Name => ImageBlocks;

        public bool Causal => false;

        public KeySpans Spans(int rows, IReadOnlyList<(int Start, int Length)> blocks, int window) => KeySpans.ImageBlocks(rows, blocks, window);
    }
}
