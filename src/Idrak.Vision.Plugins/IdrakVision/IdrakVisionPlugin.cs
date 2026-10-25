// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Plugins;
using Idrak.Models;
using Idrak.Models.Abstractions;

namespace Idrak.Vision.Plugins.IdrakVision;

/// <summary>
/// The IdrakVision plug-in: its text decoder (a Llama decoder read from <c>text_config</c>, its tensors under
/// <c>language_model.</c>, written back from a spec) in <see cref="PretrainedArchitectures"/> and its vision part in
/// <see cref="VisionFamilies"/>, both under <see cref="IdrakVisionFamily.Architecture"/>. Its image tokens attend by the
/// library's rule "image-blocks" (<see cref="ImageAttentionRules.ImageBlocks"/>: an image's tokens see each other, the
/// rest causal), or by "causal". <see cref="VisionPlugins.RegisterAll"/> registers it with the package's other families
/// (the idrak tool does at startup); an app may call <see cref="Register"/> alone.
/// </summary>
public static class IdrakVisionPlugin
{
    /// <summary>The image attention rule's name (the library's): an image's tokens attend to each other in both directions.</summary>
    public const string ImageBlocks = ImageAttentionRules.ImageBlocks;

    /// <summary>The plug-in, as <see cref="IdrakPlugins"/> records it ("idrak-vision"; listed in <see cref="VisionPlugins.All"/>).</summary>
    public static IIdrakPlugin Plugin { get; } = new IdrakVisionRegistration();

    /// <summary>Whether the family is registered.</summary>
    public static bool IsRegistered => Plugin.IsRegistered;

    /// <summary>Registers the family (its text decoder and vision part) through <see cref="IdrakPlugins"/>; once: calling it again changes nothing.</summary>
    public static void Register() => IdrakPlugins.Register(Plugin);

    /// <summary>Removes the registrations <see cref="Register"/> made, leaving the registries as it found them.</summary>
    public static void Unregister() => IdrakPlugins.Unregister(Plugin);

    // The text decoder, then the vision part.
    private sealed class IdrakVisionRegistration() : IdrakPlugin("idrak-vision")
    {
        protected override void RegisterParts()
        {
            Part(() => PretrainedArchitectures.Register(IdrakVisionFamily.Architecture, TextDecoder()), () => PretrainedArchitectures.Unregister(IdrakVisionFamily.Architecture));
            Part(() => VisionFamilies.Register(new IdrakVisionFamily()), () => VisionFamilies.Unregister(IdrakVisionFamily.Architecture));
        }
    }

    // The text side: Llama's spec from text_config, its tensors under language_model., a spec written as text_config.
    private static IPretrainedArchitecture TextDecoder() => new TextDecoderArchitecture(PretrainedFamilies.CommonArchitecture((_, spec, _) => spec, "llama"));

    private sealed class TextDecoderArchitecture(IConfigWritingArchitecture llama) : PretrainedArchitecture, IConfigWritingArchitecture
    {
        public override DecoderSpec Spec(JsonObject config, List<string> notes) => llama.Spec(config["text_config"] as JsonObject
            ?? throw new InvalidDataException($"An {IdrakVisionFamily.Architecture} config.json has a \"text_config\"."), notes);

        public override string? TensorName(string name) => PretrainedFamilies.CommonTensorName(name) is { } stored ? "language_model." + stored : null;

        public JsonObject Config(DecoderSpec spec) => new() { ["text_config"] = llama.Config(spec), ["tie_word_embeddings"] = spec.TieEmbeddings };
    }
}
