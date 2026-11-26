// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models.Abstractions;

namespace Idrak.Models;

/// <summary>
/// Reads the tokenizer of a model folder (or a tokenizer file), whatever kind it is. Finding the file's reader and
/// building the model type it names are not on the new design yet: each reader and each model's class reads and builds
/// its own.
/// </summary>
public static class PretrainedTokenizers
{
    /// <summary>
    /// The tokenizer file at <paramref name="path"/> (a model folder, or a file) and its reader: picking the reader by
    /// <see cref="TokenizerOptions.Format"/>'s name, or asking each reader, is not on the new design yet.
    /// </summary>
    /// <exception cref="NotSupportedException">Always: not on the new design yet.</exception>
    public static (string File, ITokenizerFileReader Reader)? Find(string path, TokenizerOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        return NotOnTheNewDesign.Yet<(string File, ITokenizerFileReader Reader)?>($"Finding the reader of the tokenizer at {path} among the registered ones");
    }

    /// <summary>Whether <paramref name="path"/> holds (or is) a tokenizer file a reader reads.</summary>
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
    /// <exception cref="NotSupportedException">Building the model type by its name is not on the new design yet.</exception>
    public static IPretrainedTokenizer Load(string path, TokenizerOptions? options = null)
    {
        options ??= new TokenizerOptions();
        return Create(Read(path, options));
    }

    /// <summary>The tokenizer at <paramref name="path"/> (see <see cref="Load"/>), or null when no reader finds a tokenizer file there.</summary>
    /// <exception cref="NotSupportedException">Building the model type by its name is not on the new design yet.</exception>
    public static IPretrainedTokenizer? TryLoad(string path, TokenizerOptions? options = null)
    {
        options ??= new TokenizerOptions();
        return Find(path, options) is var (file, reader) ? Create(WithConfig(reader.Read(file, options), file)) : null;
    }

    // The tokenizer of the model type a definition names: picked by that name, not on the new design yet.
    private static IPretrainedTokenizer Create(TokenizerDefinition definition) =>
        NotOnTheNewDesign.Yet<IPretrainedTokenizer>($"{definition.Source}: building the tokenizer model '{definition.ModelType}' by its name");

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

    // What a folder lacks.
    private static string Missing(string path, TokenizerOptions options) =>
        options.ResolvedFormat() is { } format ? $"No {format} tokenizer file at {path}." : $"No tokenizer file at {path}.";
}
