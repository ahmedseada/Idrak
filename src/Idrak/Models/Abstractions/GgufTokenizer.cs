// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Models.Abstractions;

/// <summary>What a GGUF tokenizer converter reads (<see cref="IGgufTokenizer"/>): the file, its tokens, and a list for notes on what it assumed.</summary>
/// <param name="File">The GGUF file.</param>
/// <param name="Tokens">Its <c>tokenizer.ggml.tokens</c>.</param>
/// <param name="Notes">Where the converter notes what it assumed (shown in the prepared model's notes).</param>
/// <param name="PreTokenizers">The pre-tokenizers handed over, a file's <c>tokenizer.ggml.pre</c> matched against their own names (null: none).</param>
public sealed record GgufTokenizerRequest(GgufFile File, IReadOnlyList<string> Tokens, List<string> Notes, IReadOnlyList<IGgufPreTokenizer>? PreTokenizers = null);

/// <summary>
/// A GGUF pre-tokenizer (a file's <c>tokenizer.ggml.pre</c>): the split pattern its family's tokenizer cuts text with
/// before byte-level BPE, and the names files call it by (llama.cpp's llama-vocab.cpp), as the family states them.
/// </summary>
public interface IGgufPreTokenizer
{
    /// <summary>The names files call it by ("qwen2", "llama-bpe").</summary>
    IReadOnlyList<string> Names { get; }

    /// <summary>The split pattern (a .NET regular expression), or null: byte-level BPE's own.</summary>
    string? Pattern { get; }
}

/// <summary>A GGUF pre-tokenizer of its names and split pattern (<see cref="IGgufPreTokenizer"/>).</summary>
/// <param name="Names">The names files call it by.</param>
/// <param name="Pattern">Its split pattern, or null: byte-level BPE's own.</param>
public sealed record GgufPreTokenizer(IReadOnlyList<string> Names, string? Pattern) : IGgufPreTokenizer;

/// <summary>
/// The converter of one GGUF tokenizer model (a file's <c>tokenizer.ggml.model</c>): the tokenizer definition of a file,
/// written as the prepared folder's tokenizer.json.
/// </summary>
public interface IGgufTokenizer
{
    /// <summary>The tokenizer definition of the file <paramref name="request"/> names.</summary>
    TokenizerDefinition Create(GgufTokenizerRequest request);

    /// <summary>The names files call this tokenizer model by (their <c>tokenizer.ggml.model</c>: "gpt2").</summary>
    IReadOnlyList<string> Names => [];
}
