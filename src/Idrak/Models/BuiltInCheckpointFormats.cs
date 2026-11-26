// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models.Abstractions;

namespace Idrak.Models;

/// <summary>A model folder with safetensors weights (one file, or shards named by model.safetensors.index.json), each tensor read as its dtype's type.</summary>
public sealed class SafeTensorsCheckpointFormat : ICheckpointFormat
{
    /// <summary>The format.</summary>
    public static ICheckpointFormat Format { get; } = new SafeTensorsCheckpointFormat();

    private SafeTensorsCheckpointFormat()
    {
    }

    /// <inheritdoc />
    public string Name => "safetensors";

    /// <inheritdoc />
    public bool CanOpen(string path) => Directory.Exists(path);

    /// <inheritdoc />
    public string Prepare(string path) => path;

    /// <inheritdoc />
    public ITensorStore Open(string folder) => SafeTensorsReader.Open(folder);

    /// <inheritdoc />
    public IEnumerable<string> Notes(string folder) => [];
}

/// <summary>
/// A .gguf file (described once by a folder in the Hugging Face layout, <see cref="GgufModel.Prepare"/>), or that folder;
/// the weights are read from the file, through the GGUF mappings of <paramref name="families"/> (the one that states the
/// file's <c>general.architecture</c>).
/// </summary>
/// <param name="families">The families whose GGUF mappings and pre-tokenizers read the files.</param>
public sealed class GgufCheckpointFormat(IFamilyProvider? families) : ICheckpointFormat
{
    /// <inheritdoc />
    public string Name => "gguf";

    /// <inheritdoc />
    public bool CanOpen(string path) =>
        File.Exists(path) && path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase) || Directory.Exists(path) && GgufModel.IsPrepared(path);

    /// <inheritdoc />
    public string Prepare(string path) => File.Exists(path) ? GgufModel.Prepare(path, families: families) : path;   // config, tokenizer and template from the file's metadata

    /// <inheritdoc />
    public ITensorStore Open(string folder) => GgufModel.OpenTensors(folder, families);

    /// <inheritdoc />
    public IEnumerable<string> Notes(string folder) => [$"weights read from {GgufModel.SourceOf(folder)}", .. GgufModel.NotesOf(folder)];
}
