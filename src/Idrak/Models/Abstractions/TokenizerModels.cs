// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Models.Abstractions;

/// <summary>
/// The tokenizer model types a tokenizer.json names (<c>model.type</c>). Building a tokenizer by the type a file names is
/// not on the new design yet: each model's class builds its own (<see cref="BpeTokenizer"/>, <c>WordPieceTokenizer</c>,
/// <c>UnigramTokenizer</c>).
/// </summary>
public static class TokenizerModels
{
    /// <summary>The library's byte-pair-encoding model.</summary>
    public const string Bpe = "BPE";

    /// <summary>The library's WordPiece model (BERT).</summary>
    public const string WordPiece = "WordPiece";

    /// <summary>The library's unigram language model (SentencePiece's: T5, ALBERT, XLNet, XLM-R).</summary>
    public const string Unigram = "Unigram";
}
