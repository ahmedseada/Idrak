// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Models;
using Idrak.Models.Abstractions;

namespace Idrak.Vision.Plugins.IdrakVision;

/// <summary>
/// The IdrakVision family's text decoder (a Llama decoder read from <c>text_config</c>, its tensors under
/// <c>language_model.</c>, written back from a spec) beside its vision part (<see cref="IdrakVisionFamily"/>), both for
/// <see cref="IdrakVisionFamily.Architecture"/>. Its image tokens attend by the library's rule "image-blocks"
/// (<see cref="ImageBlocksRule"/>: an image's tokens see each other, the rest causal), or by "causal". Both are in the
/// package's list (<see cref="BuiltInVisionFamilies"/>).
/// </summary>
public static class IdrakVisionPlugin
{
    /// <summary>The image attention rule's name (the library's): an image's tokens attend to each other in both directions.</summary>
    public const string ImageBlocks = ImageAttentionRules.ImageBlocks;

    // The text side: Llama's spec from text_config, its tensors under language_model., a spec written as text_config.
    internal static IPretrainedArchitecture TextDecoder() => new TextDecoderArchitecture(PretrainedFamilies.CommonArchitecture((_, spec, _) => spec, "llama")) { Names = [IdrakVisionFamily.Architecture] };

    private sealed class TextDecoderArchitecture(IConfigWritingArchitecture llama) : PretrainedArchitecture, IConfigWritingArchitecture
    {
        public override DecoderSpec Spec(JsonObject config, List<string> notes) => llama.Spec(config["text_config"] as JsonObject
            ?? throw new InvalidDataException($"An {IdrakVisionFamily.Architecture} config.json has a \"text_config\"."), notes);

        public override string? TensorName(string name) => PretrainedFamilies.CommonTensorName(name) is { } stored ? "language_model." + stored : null;

        public JsonObject Config(DecoderSpec spec) => new() { ["text_config"] = llama.Config(spec), ["tie_word_embeddings"] = spec.TieEmbeddings };
    }
}
