// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Models.Abstractions;

/// <summary>What a GGUF tokenizer converter reads (<see cref="IGgufTokenizer"/>): the file, its tokens, and a list for notes on what it assumed.</summary>
/// <param name="File">The GGUF file.</param>
/// <param name="Tokens">Its <c>tokenizer.ggml.tokens</c>.</param>
/// <param name="Notes">Where the converter notes what it assumed (shown in the prepared model's notes).</param>
public sealed record GgufTokenizerRequest(GgufFile File, IReadOnlyList<string> Tokens, List<string> Notes);

/// <summary>
/// The converter of one GGUF tokenizer model (a file's <c>tokenizer.ggml.model</c>): the tokenizer definition of a file,
/// written as the prepared folder's tokenizer.json.
/// </summary>
public interface IGgufTokenizer
{
    /// <summary>The tokenizer definition of the file <paramref name="request"/> names.</summary>
    TokenizerDefinition Create(GgufTokenizerRequest request);
}
