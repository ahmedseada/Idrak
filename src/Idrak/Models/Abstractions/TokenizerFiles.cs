// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Models.Abstractions;

/// <summary>
/// Reads one kind of tokenizer file (a Hugging Face tokenizer.json, a SentencePiece model, a vocabulary file) into the
/// definition a tokenizer model builds from.
/// </summary>
public interface ITokenizerFileReader
{
    /// <summary>The reader's name ("tokenizer.json"), as <see cref="TokenizerOptions.Format"/> names it.</summary>
    string Name { get; }

    /// <summary>
    /// The file this reader reads at <paramref name="path"/>: in a model folder, its file of this kind if there is one; a
    /// file, itself when it is of this kind. Null when there is none.
    /// </summary>
    string? Find(string path);

    /// <summary>Reads <paramref name="file"/> (one <see cref="Find"/> gave).</summary>
    /// <exception cref="InvalidDataException">The file is not a tokenizer of this kind.</exception>
    TokenizerDefinition Read(string file, TokenizerOptions options);
}

/// <summary>
/// The names of the tokenizer file readers (<see cref="ITokenizerFileReader.Name"/>): "tokenizer.json" (Hugging Face's,
/// any model type), "sentencepiece" (a SentencePiece model: tokenizer.model, spiece.model or sentencepiece.bpe.model) and
/// "vocab.txt" (BERT's, with tokenizer_config.json). Picking a reader by name, or asking each for a folder's file, is not
/// on the new design yet.
/// </summary>
public static class TokenizerFiles
{
    /// <summary>The Hugging Face tokenizer.json reader.</summary>
    public const string TokenizerJson = "tokenizer.json";

    /// <summary>
    /// The SentencePiece model reader (tokenizer.model, spiece.model, sentencepiece.bpe.model; a Unigram or BPE model),
    /// whose tokenizer encodes as the sentencepiece library does.
    /// </summary>
    public const string SentencePiece = "sentencepiece";

    /// <summary>The BERT-style vocab.txt reader (with tokenizer_config.json).</summary>
    public const string VocabTxt = "vocab.txt";
}
