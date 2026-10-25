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
/// knowing nothing of it: an audio encoder and its projector (Idrak.Audio's "audio" part, over its audio families). The
/// package that brings the part registers it in <see cref="ModelParts"/>; core's model loading then counts its tensors
/// as used, saving carries them (with any trained ones), a model made from a spec gets the part from its description,
/// and training opens it as a trainable soft-part source (<see cref="ITrainableSoftParts"/>).
/// </summary>
public interface IModelPart
{
    /// <summary>The part's kind: also the <see cref="ChatPart.Kind"/> its source reads ("audio").</summary>
    string Kind { get; }

    /// <summary>Whether a config.json holds this part (an "audio_config", say).</summary>
    bool In(JsonObject config);

    /// <summary>The checkpoint tensors (by their stored names) the part owns, the shapes checked.</summary>
    /// <exception cref="NotSupportedException">No family of the part is registered for the architecture (the message names its registry).</exception>
    /// <exception cref="InvalidDataException">A tensor is missing or its shape disagrees with the part's configuration.</exception>
    IReadOnlySet<string> StoredTensors(ModelPartCheckpoint checkpoint);

    /// <summary>The part for a model made from a spec: its tensors drawn from the request's seed, its config keys and files.</summary>
    /// <exception cref="NotSupportedException">The part's family cannot make it from a description.</exception>
    CreatedModelPart Create(ModelPartRequest request);

    /// <summary>The part as a soft-part source on <paramref name="options"/>' device, its trained tensors put in. Dispose it when done.</summary>
    ITrainableSoftParts Open(ModelPartCheckpoint checkpoint, ModelPartOptions options);
}

/// <summary>
/// The model parts, by kind. The library itself registers none in this package: Idrak.Audio registers "audio" when the
/// application ships it (a library default, through the packages' registrations). A checkpoint holding a part whose kind
/// is not registered loads as its decoder alone, its tensors counted unused. Register another with <see cref="Register"/>.
/// </summary>
public static class ModelParts
{
    private static readonly SlotTable<string, IModelPart> Table = new(nameof(ModelParts), comparer: StringComparer.Ordinal,
        unguarded: "a part's tensors, files and source must agree with one another, so two registrations cannot stand in for each other");

    static ModelParts() => LibraryDefaults.Ensure(typeof(ModelParts));   // the packages' parts (Idrak.Audio's "audio"), on first use

    /// <summary>Registers <paramref name="part"/> under its <see cref="IModelPart.Kind"/>; under a library kind it takes that part's place until <see cref="Unregister"/>.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IModelPart part)
    {
        ArgumentNullException.ThrowIfNull(part);
        ArgumentException.ThrowIfNullOrEmpty(part.Kind);
        Table.Register(part.Kind, part, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's part <paramref name="kind"/> (a library kind gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string kind) => Table.Unregister(kind);

    /// <summary>The registered kinds.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The part registered as <paramref name="kind"/>, or null.</summary>
    public static IModelPart? Find(string kind) => Table.Find(kind);

    /// <summary>The part registered as <paramref name="kind"/>.</summary>
    /// <exception cref="NotSupportedException">None is; the message names those that are and <see cref="Register"/>.</exception>
    public static IModelPart Get(string kind) => Table.Find(kind)
        ?? throw new NotRegisteredException(nameof(ModelParts), kind, Table.Keys,
            $"Model part '{kind}' is not registered ({(Table.Keys.Count == 0 ? "none is" : string.Join(", ", Table.Keys))}); "
            + "ship the package that brings it (Idrak.Audio for \"audio\") or add it with ModelParts.Register.");

    /// <summary>The library's part <paramref name="kind"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IModelPart? Default(string kind) => Table.Default(kind);

    /// <summary>Who registered the part <paramref name="kind"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string kind) => Table.Origin(kind);

    /// <summary>The registered parts a config.json holds, in registration order.</summary>
    public static IReadOnlyList<IModelPart> In(JsonObject config)
    {
        ArgumentNullException.ThrowIfNull(config);
        var found = new List<IModelPart>();
        foreach (string kind in Table.Keys)
        {
            if (Table.Find(kind) is { } part && part.In(config))
            {
                found.Add(part);
            }
        }

        return found;
    }
}
