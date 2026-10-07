// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

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
    private static readonly List<ICheckpointFormat> Registry = [];

    static CheckpointFormats() => LibraryModelFormats.RegisterCheckpointFormats();   // the built-in formats, on first use

    /// <summary>
    /// Registers <paramref name="format"/>: it replaces the format of the same name (in its place), or is asked before
    /// every format registered so far.
    /// </summary>
    public static void Register(ICheckpointFormat format)
    {
        ArgumentNullException.ThrowIfNull(format);
        lock (Registry)
        {
            int at = Registry.FindIndex(f => f.Name == format.Name);
            if (at >= 0)
            {
                Registry[at] = format;
            }
            else
            {
                Registry.Insert(0, format);
            }
        }
    }

    /// <summary>Removes the format registered as <paramref name="name"/>; false when there is none.</summary>
    public static bool Unregister(string name)
    {
        lock (Registry)
        {
            return Registry.RemoveAll(f => f.Name == name) > 0;
        }
    }

    /// <summary>The registered format names, in the order they are asked.</summary>
    public static IReadOnlyCollection<string> Names
    {
        get
        {
            lock (Registry)
            {
                return [.. Registry.Select(f => f.Name)];
            }
        }
    }

    /// <summary>The format registered as <paramref name="name"/>.</summary>
    public static ICheckpointFormat Get(string name)
    {
        lock (Registry)
        {
            return Registry.Find(f => f.Name == name)
                ?? throw new NotSupportedException($"No checkpoint format '{name}' is registered ({string.Join(", ", Registry.Select(f => f.Name))}); add it with CheckpointFormats.Register.");
        }
    }

    /// <summary>The format that reads <paramref name="path"/> (the first registered one whose <see cref="ICheckpointFormat.CanOpen"/> says so).</summary>
    public static ICheckpointFormat For(string path)
    {
        ICheckpointFormat[] formats;
        lock (Registry)
        {
            formats = [.. Registry];
        }

        // Asked outside the lock: a format's CanOpen may look at the disk.
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
