// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
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
