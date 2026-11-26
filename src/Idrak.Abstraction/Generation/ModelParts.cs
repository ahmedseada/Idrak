// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Generation;

/// <summary>A model folder's checkpoint as a model part reads it.</summary>
/// <param name="Architecture">The architecture name the model was matched by (config.json's "architectures").</param>
/// <param name="Config">The model's config.json.</param>
/// <param name="Tensors">The checkpoint's tensors by their stored names, open while the call runs (do not keep it).</param>
/// <param name="Folder">The model folder (where the part's other files are), or null.</param>
/// <param name="Open">Opens the checkpoint again, for reading later (the caller disposes what it returns).</param>
/// <param name="Notes">Append anything approximated (shown in the loaded model's notes).</param>
public sealed record ModelPartCheckpoint(string Architecture, JsonObject Config, ITensorStore Tensors, string? Folder, Func<ITensorStore> Open, List<string> Notes);

/// <summary>What a model made from a spec asks of a part (<see cref="IModelPart.Create"/>).</summary>
/// <param name="Architecture">The model's architecture name (the family the part is matched by).</param>
/// <param name="Description">The part's own description, as the spec gives it (the library does not read it).</param>
/// <param name="TextDim">The text decoder's width (each soft token's embedding).</param>
/// <param name="Tokenizer">The model's tokenizer (the part's tokens are looked up in it).</param>
/// <param name="Seed">The seed the part's weights are drawn from.</param>
public sealed record ModelPartRequest(string Architecture, JsonObject Description, int TextDim, ITokenizer Tokenizer, int Seed);

/// <summary>A part made for a new model: what is written into its folder so the part reads it back like any checkpoint.</summary>
/// <param name="Config">Keys for config.json's top level (the part's configuration and token ids).</param>
/// <param name="Tensors">The part's tensors under the checkpoint's names, as the checkpoint stores them.</param>
/// <param name="Files">Other files of the model folder by name, with their text (a preprocessor configuration, say).</param>
public sealed record CreatedModelPart(JsonObject Config, IReadOnlyList<(string Name, int[] Shape, float[] Values)> Tensors, IReadOnlyDictionary<string, string> Files);

/// <summary>How a model part's soft-part source is opened (<see cref="IModelPart.Open"/>).</summary>
public sealed record ModelPartOptions
{
    /// <summary>Where it runs (the language model's device).</summary>
    public required Device Device { get; init; }

    /// <summary>Trained tensors by their checkpoint names, replacing the checkpoint's (a training run's kept parts); none when null.</summary>
    public IReadOnlyDictionary<string, (int[] Shape, float[] Values)>? Trained { get; init; }

    /// <summary>Where the part keeps what it computed for a part it saw before (frozen stages' features); none when null.</summary>
    public IFeatureCache? Cache { get; init; }
}

/// <summary>
/// A part of a model stored beside its text decoder in the same folder and checkpoint, the decoder's own architecture
/// knowing nothing of it: an audio encoder and its projector (Idrak.Audio's "audio" part, over its audio families).
/// </summary>
public interface IModelPart
{
    /// <summary>The part's kind: also the <see cref="ChatPart.Kind"/> its source reads ("audio").</summary>
    string Kind { get; }

    /// <summary>Whether a config.json holds this part (an "audio_config", say).</summary>
    bool In(JsonObject config);

    /// <summary>The checkpoint tensors (by their stored names) the part owns, the shapes checked.</summary>
    /// <exception cref="NotSupportedException">The part's family for the architecture cannot be picked (by its name: not on the new design yet).</exception>
    /// <exception cref="InvalidDataException">A tensor is missing or its shape disagrees with the part's configuration.</exception>
    IReadOnlySet<string> StoredTensors(ModelPartCheckpoint checkpoint);

    /// <summary>The part for a model made from a spec: its tensors drawn from the request's seed, its config keys and files.</summary>
    /// <exception cref="NotSupportedException">The part's family cannot make it from a description.</exception>
    CreatedModelPart Create(ModelPartRequest request);

    /// <summary>The part as a soft-part source on <paramref name="options"/>' device, its trained tensors put in. Dispose it when done.</summary>
    ITrainableSoftParts Open(ModelPartCheckpoint checkpoint, ModelPartOptions options);
}

