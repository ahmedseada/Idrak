// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;

namespace Idrak.Models.Abstractions;

/// <summary>
/// The vision part of a vision-language checkpoint as its family reads it (<see cref="IVisionFamily.Read"/>): everything
/// the rest of the library needs to put images into the prompt, and nothing about how the family builds its encoder. A
/// family derives from it and adds what is its own (Gemma 3's <c>Gemma3Vision</c>: SigLIP's configuration, the projector's
/// pooling, the image token ids). The loaded model carries it (<c>PretrainedModel.Vision</c>).
/// </summary>
public abstract class PretrainedVision
{
    /// <summary>The family it was read by: its name in <see cref="VisionFamilies"/> (the architecture name).</summary>
    public abstract string Family { get; }

    /// <summary>The width of the image features: the text decoder's (one embedding per image token).</summary>
    public abstract int Width { get; }

    /// <summary>How the rendered prompt's image markers expand to the image tokens, with the family's token ids.</summary>
    public abstract IImagePromptFormat PromptFormat { get; }

    /// <summary>How the image tokens attend in the prompt (a rule of <see cref="ImageAttentionRules"/>, the family's choice).</summary>
    public abstract IImageAttentionRule Attention { get; }

    /// <summary>The checkpoint tensors (by their stored names) the vision part reads: they do not count as unused.</summary>
    public abstract IReadOnlyCollection<string> StoredTensors { get; }

    /// <summary>
    /// Builds the family's vision encoder (with its own preprocessing) from the checkpoint: decoded images in, each
    /// image's features out (<see cref="IVisionEncoder"/>). Dispose it when done.
    /// </summary>
    public abstract IVisionEncoder CreateEncoder(VisionEncoderOptions? options = null);

    /// <summary>
    /// The keys of the <see cref="VisionOptions"/> this family's encoder takes (at creation,
    /// <see cref="VisionEncoderOptions.VisionOptions"/>, and per request); none unless the family says. Callers check
    /// options against them early (<see cref="VisionOptions.ThrowIfUnknown"/>); the encoder checks the values.
    /// </summary>
    public virtual IReadOnlyCollection<string> VisionOptionKeys => [];

    /// <summary>One line for <c>idrak show</c> and messages: the family's encoder and image tokens.</summary>
    public virtual string Describe() => $"{Family}, {PromptFormat.Name}, image tokens attend: {Attention.Name}";

    /// <inheritdoc />
    public override string ToString() => Describe();
}

/// <summary>How <see cref="PretrainedVision.CreateEncoder"/> builds an encoder.</summary>
public sealed record VisionEncoderOptions
{
    /// <summary>Where it runs (<see cref="Device.Default"/> when null; pass the language model's device).</summary>
    public Device? Device { get; init; }

    /// <summary>Turn images to grayscale before the family's preprocessing (some fine-tunes ask for it).</summary>
    public bool Grayscale { get; init; }

    /// <summary>
    /// The family's own options for every image the encoder reads (its keys, <see cref="PretrainedVision.VisionOptionKeys"/>),
    /// over the family's defaults from the checkpoint's files; a request's options go over these per key. Null: none.
    /// </summary>
    /// <remarks>The family refuses a key it does not take, naming those it does, when it builds the encoder.</remarks>
    public VisionOptions? VisionOptions { get; init; }

    /// <summary>
    /// How the encoder holds its projection weights: as the checkpoint stores them (the default: a bfloat16 checkpoint's
    /// weights stay bfloat16, half the memory of float32 for the same values, while inputs, outputs and sums stay
    /// float32), all float32, or bfloat16 (rounded when stored wider). A family builds its layers with
    /// <c>StoredWeights</c> to honour it.
    /// </summary>
    public EncoderWeights Weights { get; init; }
}

/// <summary>How an encoder built from a checkpoint holds its projection weights (<see cref="VisionEncoderOptions.Weights"/>).</summary>
public enum EncoderWeights
{
    /// <summary>
    /// As the checkpoint stores them (<see cref="ITensorStore.FormatOf"/>): bfloat16 weights as bfloat16, anything else
    /// as float32. The computation is the float32 one either way (the same weight values, float32 activations and sums).
    /// </summary>
    AsStored,

    /// <summary>Every weight in float32 (twice the memory of a bfloat16 checkpoint's, the same values).</summary>
    Float32,

    /// <summary>Projection weights in bfloat16, rounded to nearest when the checkpoint stores them wider.</summary>
    BFloat16,
}

/// <summary>What a vision family reads a checkpoint's vision part from (<see cref="IVisionFamily.Read"/>).</summary>
/// <param name="Architecture">The architecture name the checkpoint was matched by.</param>
/// <param name="Config">The model's config.json.</param>
/// <param name="Tensors">The checkpoint's tensors by their stored names, open while reading (do not keep it).</param>
/// <param name="Folder">The model folder (where <c>preprocessor_config.json</c> and the like are), or null.</param>
/// <param name="Open">Opens the checkpoint again, for building the encoder later (the caller disposes what it returns).</param>
/// <param name="Notes">Append anything approximated (shown in the loaded model's notes).</param>
public sealed record VisionCheckpoint(string Architecture, JsonObject Config, ITensorStore Tensors, string? Folder, Func<ITensorStore> Open, List<string> Notes);

/// <summary>
/// A vision-language family's image side: how its checkpoints' vision part is read. Registered in
/// <see cref="VisionFamilies"/> under the architecture name its checkpoints give (config.json's "architectures"), beside the
/// text decoder registered under the same name in <see cref="PretrainedArchitectures"/>.
/// </summary>
public interface IVisionFamily
{
    /// <summary>The architecture name it is registered under.</summary>
    string Name { get; }

    /// <summary>
    /// The checkpoint's vision part (configuration, tensors checked, encoder, prompt format, attention rule), or null when
    /// this checkpoint of the family has no vision weights (a note says so).
    /// </summary>
    /// <exception cref="InvalidDataException">A tensor is missing or its shape disagrees with the configuration.</exception>
    PretrainedVision? Read(VisionCheckpoint checkpoint);
}

/// <summary>
/// The vision-language families, by architecture name (config.json's "architectures"). The library registers none: a
/// family (its configuration, tensor names, encoder, preprocessing, prompt format, token ids and attention rule) is an
/// application of these contracts, registered by the app or plug-in that brings it (the Gemma 3 and LLaVA registrations
/// are samples, <c>samples/Gemma3Vision</c> and <c>tests/Idrak.PluginTests</c>). A checkpoint is matched to its
/// own family by that exact name; a checkpoint with a vision part (a <c>vision_config</c>) whose name has no family is
/// refused, naming this registry: nothing falls back to another family, since families are not interchangeable. Register
/// one with <see cref="Register"/> (and its text decoder with <see cref="PretrainedArchitectures.Register"/>).
/// </summary>
public static class VisionFamilies
{
    private static readonly SlotTable<string, IVisionFamily> Registry = new(nameof(VisionFamilies), comparer: StringComparer.Ordinal,
        unguarded: "a family's reading, encoder, prompt format and attention rule must agree with one another, so two registrations cannot be mixed");

    /// <summary>Registers <paramref name="family"/> under its <see cref="IVisionFamily.Name"/>; under a library name it takes that family's place until <see cref="Unregister"/>.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IVisionFamily family)
    {
        ArgumentNullException.ThrowIfNull(family);
        ArgumentException.ThrowIfNullOrEmpty(family.Name);
        Registry.Register(family.Name, family, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's family <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered architecture names.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The family registered as <paramref name="name"/>, or null.</summary>
    public static IVisionFamily? Find(string name) => Registry.Find(name);

    /// <summary>The family registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is; the message names <see cref="Register"/>.</exception>
    public static IVisionFamily Get(string name) => Registry.Find(name) ?? throw NotRegistered(name);

    /// <summary>The library's family <paramref name="name"/>, whatever an app registered over it (for an app's family to build on); null when the library has none.</summary>
    public static IVisionFamily? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the family <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>Whether a config.json describes a vision-language checkpoint: it has a <c>vision_config</c>.</summary>
    public static bool HasVision(JsonObject config)
    {
        ArgumentNullException.ThrowIfNull(config);
        return config["vision_config"] is JsonObject;
    }

    /// <summary>
    /// The family that reads the vision part of a checkpoint of architecture <paramref name="architecture"/> with this
    /// config.json: the one registered under that exact name, or null for a checkpoint with no vision part.
    /// </summary>
    /// <exception cref="NotSupportedException">The checkpoint has a vision part and no family is registered under its name; the message names this registry.</exception>
    public static IVisionFamily? For(string architecture, JsonObject config)
    {
        ArgumentException.ThrowIfNullOrEmpty(architecture);
        ArgumentNullException.ThrowIfNull(config);
        return Registry.Find(architecture) ?? (HasVision(config) ? throw NotRegistered(architecture) : null);
    }

    internal static NotSupportedException NotRegistered(string name) =>
        new($"Vision family '{name}' is not registered (registered: {(Registry.Keys.Count == 0 ? "none" : string.Join(", ", Registry.Keys))}); "
            + "register it with VisionFamilies.Register (a plug-in or the app that brings the family; the library registers none).");
}
