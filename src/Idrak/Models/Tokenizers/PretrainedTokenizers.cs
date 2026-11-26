// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models.Abstractions;

namespace Idrak.Models;

/// <summary>
/// Reads the tokenizer of a model folder (or a tokenizer file): a Hugging Face tokenizer.json, reached directly, its model
/// built from the file's own model type (tokenizer.json's table: BPE, Unigram, WordPiece). Finding any other file's
/// reader is not on the new design yet.
/// </summary>
public static class PretrainedTokenizers
{
    /// <summary>
    /// The tokenizer file at <paramref name="path"/> (a model folder, or a file) and its reader: a tokenizer.json, read by
    /// the tokenizer.json reader directly; null where there is none. Picking another reader by
    /// <see cref="TokenizerOptions.Format"/>'s name, or asking each reader, is not on the new design yet.
    /// </summary>
    /// <exception cref="NotSupportedException">The options name another format: not on the new design yet.</exception>
    public static (string File, ITokenizerFileReader Reader)? Find(string path, TokenizerOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        var json = new TokenizerJsonReader();
        if (options?.ResolvedFormat() is { } format && format != json.Name)
        {
            return NotOnTheNewDesign.Yet<(string File, ITokenizerFileReader Reader)?>($"Finding the reader of the {format} tokenizer at {path}");
        }

        return json.Find(path) is { } file ? (file, json) : null;
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
        return Create(Read(path, options), options);
    }

    /// <summary>The tokenizer at <paramref name="path"/> (see <see cref="Load"/>), or null when no reader finds a tokenizer file there.</summary>
    /// <exception cref="NotSupportedException">Building the model type by its name is not on the new design yet.</exception>
    public static IPretrainedTokenizer? TryLoad(string path, TokenizerOptions? options = null)
    {
        options ??= new TokenizerOptions();
        return Find(path, options) is var (file, reader) ? Create(WithConfig(reader.Read(file, options), file), options) : null;
    }

    // The tokenizer of the model type a tokenizer.json names (the format's own table of its model types).
    private static IPretrainedTokenizer Create(TokenizerDefinition definition, TokenizerOptions options) => definition.ModelType switch
    {
        "BPE" => BpeTokenizer.FromDefinition(definition, options),
        "Unigram" => UnigramTokenizer.FromDefinition(definition, options),
        "WordPiece" => WordPieceTokenizer.FromDefinition(definition, options),
        _ => NotOnTheNewDesign.Yet<IPretrainedTokenizer>($"{definition.Source}: building the tokenizer model '{definition.ModelType}'"),
    };

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
