// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Models.Abstractions;

/// <summary>What a vision family reads a checkpoint's vision part from (<see cref="IVisionFamily.Read"/>).</summary>
/// <param name="Architecture">The architecture name the checkpoint was matched by.</param>
/// <param name="Config">The model's config.json.</param>
/// <param name="Tensors">The checkpoint's tensors by their stored names, open while reading (do not keep it).</param>
/// <param name="Folder">The model folder (where <c>preprocessor_config.json</c> and the like are), or null.</param>
/// <param name="Open">Opens the checkpoint again, for building the encoder later (the caller disposes what it returns).</param>
/// <param name="Notes">Append anything approximated (shown in the loaded model's notes).</param>
public sealed record VisionCheckpoint(string Architecture, JsonObject Config, ITensorStore Tensors, string? Folder, Func<ITensorStore> Open, List<string> Notes);

/// <summary>
/// A vision-language family's image side: how its checkpoints' vision part is read, beside its text decoder
/// (<see cref="IPretrainedArchitecture"/>).
/// </summary>
public interface IVisionFamily
{
    /// <summary>The architecture name its checkpoints give (config.json's "architectures").</summary>
    string Name { get; }

    /// <summary>
    /// The names its checkpoints call it by (config.json's "architectures"), the family's own: a checkpoint's name is matched
    /// against them where it comes in (<see cref="IFamilyProvider.VisionFamilies"/>, walked in order). <see cref="Name"/> by default.
    /// </summary>
    IReadOnlyList<string> Names => [Name];

    /// <summary>
    /// The checkpoint's vision part (configuration, tensors checked, encoder, prompt format, attention rule), or null when
    /// this checkpoint of the family has no vision weights (a note says so).
    /// </summary>
    /// <exception cref="InvalidDataException">A tensor is missing or its shape disagrees with the configuration.</exception>
    IPretrainedVision? Read(VisionCheckpoint checkpoint);

    /// <summary>
    /// A vision part that did not exist, for a model made from a spec (<c>PretrainedModel.Create</c> with
    /// <c>CreateOptions.Vision</c>): its checkpoint tensors drawn from <paramref name="seed"/>, the config.json keys that
    /// describe it (<see cref="Read"/> reads them back) and its other files. <paramref name="vision"/> is the family's own
    /// description of the part (sizes, token names, attention, preprocessing): the library does not read it. A family
    /// read from checkpoints only keeps this default, which refuses.
    /// </summary>
    /// <param name="vision">The family's description of the part.</param>
    /// <param name="textDim">The text decoder's width (each image token's embedding).</param>
    /// <param name="tokenizer">The model's tokenizer (the family's image tokens are looked up in it).</param>
    /// <param name="seed">The seed the weights are drawn from.</param>
    /// <exception cref="NotSupportedException">The family cannot make a vision part from a description.</exception>
    /// <exception cref="InvalidDataException">The description is wrong (a size, a token the tokenizer lacks, …).</exception>
    CreatedVision Create(JsonObject vision, int textDim, ITokenizer tokenizer, int seed) =>
        throw new NotSupportedException($"Vision family '{Name}' cannot make a vision part from a description (it reads checkpoints only).");
}

/// <summary>
/// A vision part made by <see cref="IVisionFamily.Create"/>: what is written into the model folder so the family's
/// <see cref="IVisionFamily.Read"/> reads it like any checkpoint.
/// </summary>
/// <param name="Config">Keys for config.json's top level (the family's vision configuration and image token ids).</param>
/// <param name="Tensors">The part's tensors under the checkpoint's names, as the checkpoint stores them.</param>
/// <param name="Files">Other files of the model folder by name, with their text (preprocessor_config.json, for example).</param>
public sealed record CreatedVision(JsonObject Config, IReadOnlyList<(string Name, int[] Shape, float[] Values)> Tensors, IReadOnlyDictionary<string, string> Files);
