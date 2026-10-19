// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Models.Abstractions;

/// <summary>
/// A tokenizer as a reader (<see cref="TokenizerFiles"/>) found it, for its model (<see cref="TokenizerModels"/>) to build:
/// the JSON of a tokenizer.json without the model's large tables, and those tables read straight from the file's bytes
/// (a quarter of a million vocabulary entries as one dictionary or one array, not as a JSON tree). Every reader gives this
/// one shape, so a tokenizer.json, a SentencePiece model and a vocabulary file of the same tokenizer build the same one.
/// <para>
/// A model may take the tables as its own (the BPE model adds the added tokens to <see cref="Vocabulary"/>): a definition
/// builds one tokenizer.
/// </para>
/// </summary>
public sealed class TokenizerDefinition
{
    /// <summary>The file it was read from (or what it was built from), for messages.</summary>
    public required string Source { get; init; }

    /// <summary>
    /// The tokenizer.json: <c>normalizer</c>, <c>pre_tokenizer</c>, <c>post_processor</c>, <c>decoder</c>,
    /// <c>added_tokens</c> and <c>model</c> without its <c>vocab</c> and <c>merges</c>.
    /// </summary>
    public required JsonObject Json { get; init; }

    /// <summary>The model's vocabulary when tokenizer.json writes it as an object of token to id (BPE, WordPiece); else null.</summary>
    public Dictionary<string, int>? Vocabulary { get; init; }

    /// <summary>The model's vocabulary when tokenizer.json writes it as a list of [piece, score] (Unigram; id = position); else null.</summary>
    public IReadOnlyList<(string Piece, double Score)>? Pieces { get; init; }

    /// <summary>The model's merges in rank order (BPE); else null.</summary>
    public (string Left, string Right)[]? Merges { get; init; }

    /// <summary>The model's type (<c>model.type</c>: "BPE", "Unigram", "WordPiece", …), or "" when the file names none.</summary>
    public string ModelType => Json["model"]?["type"] is JsonValue type && type.TryGetValue(out string? name) ? name : "";

    // A copy whose tables a second model may take (the library's model, when an app's model of the same type failed or is
    // compared with it): the JSON and the vocabulary are copied, the read-only tables shared.
    internal TokenizerDefinition Copy() => new()
    {
        Source = Source,
        Json = (JsonObject)Json.DeepClone(),
        Vocabulary = Vocabulary is null ? null : new Dictionary<string, int>(Vocabulary, Vocabulary.Comparer),
        Pieces = Pieces,
        Merges = Merges,
    };
}
