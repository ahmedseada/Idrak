// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Models.Abstractions;

/// <summary>
/// A way pretrained weights are stored. Loading a pretrained model (Idrak's <c>PretrainedModel.Load</c>) asks
/// the registered formats (see <see cref="CheckpointFormats"/>) which one reads the path it is given; that one turns the
/// path into a folder in the Hugging Face layout (config.json, tokenizer files, chat template) and opens the weights under
/// Hugging Face names, which the architecture (<see cref="PretrainedArchitecture.TensorName"/>) then maps to Idrak's.
/// </summary>
public interface ICheckpointFormat
{
    /// <summary>The format's name ("safetensors", "gguf", …): registering another format under it replaces this one.</summary>
    string Name { get; }

    /// <summary>
    /// Whether this format reads <paramref name="path"/>: a file or a folder as the user named it, or a folder
    /// <see cref="Prepare"/> returned before.
    /// </summary>
    bool CanOpen(string path);

    /// <summary>
    /// The folder in the Hugging Face layout for <paramref name="path"/> (at least config.json): <paramref name="path"/>
    /// itself when it is one, else a folder made from it (GGUF writes one from the file's metadata, once).
    /// </summary>
    string Prepare(string path);

    /// <summary>The weights of the folder <see cref="Prepare"/> returned, by Hugging Face name.</summary>
    ITensorStore Open(string folder);

    /// <summary>What the model's notes should say about this checkpoint (where the weights come from, fallbacks taken).</summary>
    IEnumerable<string> Notes(string folder);
}

/// <summary>
/// The checkpoint formats pretrained models are loaded from. Idrak registers GGUF (a .gguf file, or the
/// folder it prepared for one) and safetensors (a model folder); add others with <see cref="Register"/>. A path is read by
/// the first format that can open it, the most recently registered first, so a new format can claim paths a built-in one
/// would also take.
/// </summary>
public static class CheckpointFormats
{
    // In the order they are asked.
    private static readonly SlotTable<string, ICheckpointFormat> Registry = new(nameof(CheckpointFormats), (slot, app, library) => new GuardedFormat(slot, app, library),
        StringComparer.Ordinal, newestFirst: true);

    static CheckpointFormats() => Overrides.AsLibraryDefaults(LibraryModelFormats.RegisterCheckpointFormats);   // the built-in formats, on first use

    /// <summary>
    /// Registers <paramref name="format"/>: it takes the place of the format of the same name (the library's stays behind
    /// it, see <see cref="SetPolicy"/>), or is asked before every format registered so far.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ICheckpointFormat format)
    {
        ArgumentNullException.ThrowIfNull(format);
        Registry.Register(format.Name, format, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's format <paramref name="name"/> (a built-in name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered format names, in the order they are asked.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The format registered as <paramref name="name"/>.</summary>
    public static ICheckpointFormat Get(string name) => Registry.Find(name)
        ?? throw new NotSupportedException($"No checkpoint format '{name}' is registered ({string.Join(", ", Registry.Keys)}); add it with CheckpointFormats.Register.");

    /// <summary>The library's format <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static ICheckpointFormat? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the format <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>
    /// What happens when the app's format <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set: the error reaches the caller;
    /// <see cref="SlotPolicy.FallBack"/> retries on the library's): each call falls back on its own. Under <see cref="SlotPolicy.Shadow"/> only <see cref="ICheckpointFormat.CanOpen"/>
    /// and <see cref="ICheckpointFormat.Notes"/> are compared: preparing writes files and opening holds them, which is not done twice.
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Registry.SetPolicy(name, policy, shadowRate);

    private sealed class GuardedFormat(Slot slot, ICheckpointFormat app, ICheckpointFormat library) : ICheckpointFormat
    {
        public string Name => app.Name;

        public bool CanOpen(string path) => slot.Call(() => app.CanOpen(path), () => library.CanOpen(path), Comparisons.Exact);

        public string Prepare(string path) => slot.Call(() => app.Prepare(path), () => library.Prepare(path), effects: true);

        public ITensorStore Open(string folder) => slot.Call(() => app.Open(folder), () => library.Open(folder), effects: true);

        public IEnumerable<string> Notes(string folder) =>
            slot.Call<IReadOnlyList<string>>(() => [.. app.Notes(folder)], () => [.. library.Notes(folder)], Comparisons.Sequences);
    }

    /// <summary>The format that reads <paramref name="path"/> (the first registered one whose <see cref="ICheckpointFormat.CanOpen"/> says so).</summary>
    public static ICheckpointFormat For(string path)
    {
        var formats = Registry.Values;
        foreach (var format in formats)
        {
            if (format.CanOpen(path))
            {
                return format;
            }
        }

        if (!File.Exists(path) && !Directory.Exists(path))
        {
            throw new DirectoryNotFoundException($"{path} does not exist.");
        }

        throw new NotSupportedException($"No checkpoint format reads {path} ({string.Join(", ", formats.Select(f => f.Name))}); add one with CheckpointFormats.Register.");
    }
}
