// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models.Abstractions;

namespace Idrak.Models;

/// <summary>
/// Reads the tokenizer of a model folder (or a tokenizer file), whatever kind it is: a registered reader
/// (<see cref="TokenizerFiles"/>) finds and reads the file, and the model type it names (<see cref="TokenizerModels"/>)
/// builds the tokenizer. Nothing here knows a model type: BPE and the others are registered ones.
/// </summary>
public static class PretrainedTokenizers
{
    /// <summary>
    /// The tokenizer file at <paramref name="path"/> (a model folder, or a file) and its reader: the reader
    /// <see cref="TokenizerOptions.Format"/> names, else the first registered reader that finds its file. Null when none
    /// does.
    /// </summary>
    /// <exception cref="NotSupportedException">The format the options name is not registered.</exception>
    public static (string File, ITokenizerFileReader Reader)? Find(string path, TokenizerOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if ((options ?? new TokenizerOptions()).ResolvedFormat() is { } format)
        {
            var reader = TokenizerFiles.Get(format);
            return reader.Find(path) is { } file ? (file, reader) : null;
        }

        foreach (var reader in TokenizerFiles.Readers)
        {
            if (reader.Find(path) is { } file)
            {
                return (file, reader);
            }
        }

        return null;
    }

    /// <summary>Whether <paramref name="path"/> holds (or is) a tokenizer file a registered reader reads.</summary>
    public static bool Exists(string path, TokenizerOptions? options = null) => Find(path, options) is not null;

    /// <summary>The definition of the tokenizer at <paramref name="path"/>, as its reader found it (see <see cref="Find"/>).</summary>
    /// <exception cref="FileNotFoundException">No reader finds a tokenizer file there.</exception>
    public static TokenizerDefinition Read(string path, TokenizerOptions? options = null)
    {
        options ??= new TokenizerOptions();
        var (file, reader) = Find(path, options) ?? throw new FileNotFoundException(Missing(path, options), path);
        return WithConfig(reader.Read(file, options), file);
    }

    /// <summary>The tokenizer at <paramref name="path"/> (a model folder, or a tokenizer file), built by its model type.</summary>
    /// <exception cref="FileNotFoundException">No reader finds a tokenizer file there.</exception>
    /// <exception cref="NotSupportedException">The file's model type is not registered.</exception>
    public static IPretrainedTokenizer Load(string path, TokenizerOptions? options = null)
    {
        options ??= new TokenizerOptions();
        return TokenizerModels.Create(Read(path, options), options);
    }

    /// <summary>The tokenizer at <paramref name="path"/> (see <see cref="Load"/>), or null when no reader finds a tokenizer file there.</summary>
    /// <exception cref="NotSupportedException">The file's model type is not registered.</exception>
    public static IPretrainedTokenizer? TryLoad(string path, TokenizerOptions? options = null)
    {
        options ??= new TokenizerOptions();
        return Find(path, options) is var (file, reader) ? TokenizerModels.Create(WithConfig(reader.Read(file, options), file), options) : null;
    }

    // What the model's tokenizer_config.json beside the file says of the decoded text's spaces (clean_up_tokenization_spaces),
    // unless the reader set it.
    private static TokenizerDefinition WithConfig(TokenizerDefinition definition, string file)
    {
        string config = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(file)) ?? ".", "tokenizer_config.json");
        if (definition.CleanUpTokenizationSpaces is null && File.Exists(config))
        {
            System.Text.Json.Nodes.JsonNode? json;
            try
            {
                json = System.Text.Json.Nodes.JsonNode.Parse(File.ReadAllText(config));
            }
            catch (System.Text.Json.JsonException e)
            {
                throw new InvalidDataException($"{config} is not JSON: {e.Message}", e);
            }

            if (json?["clean_up_tokenization_spaces"] is System.Text.Json.Nodes.JsonValue value && value.TryGetValue(out bool clean))
            {
                definition.CleanUpTokenizationSpaces = clean;
            }
        }

        return definition;
    }

    // What a folder lacks: the files the registered readers look for.
    private static string Missing(string path, TokenizerOptions options) =>
        (options.ResolvedFormat() is { } format ? $"No {format} tokenizer file at {path}" : $"No tokenizer file at {path}")
        + $" (readers: {string.Join(", ", TokenizerFiles.Names)}); add a reader with TokenizerFiles.Register.";
}
