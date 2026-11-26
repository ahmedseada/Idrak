// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Formats;

/// <summary>
/// A way pretrained weights are stored.
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

