// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Models;
using Idrak.Models.Abstractions;

// Plan 24, phase T2: WordPiece (BERT) against Hugging Face tokenizers.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] TokenizerWordPieceGroup =
    [
        ("tokenizers: wordpiece: lowercase is Rust's (İ becomes i and a combining dot) in Lowercase and BertNormalizer, string and span forms alike", TokenizersLowercaseDottedI),
    ];

    internal static void TokenizersLowercaseDottedI(Device device)
    {
        Check(BpeTokenizer.NormalizerStep(JsonNode.Parse("""{"type": "Lowercase"}""")!.AsObject())("İSTANBUL ΣΑΣ") == "i\u0307stanbul σασ", "Lowercase (string)");
        using var output = new PreTokenizedText();
        new TokenizerPipeline(JsonNode.Parse("""{"normalizer": {"type": "Lowercase"}}""")!.AsObject()).Run("İSTANBUL", 0, 8, true, output);
        Check(new string(output.Piece(0)) == "i\u0307stanbul" && output.Text.OriginOf(0) == (0, 1) && output.Text.OriginOf(1) == (0, 1),
            "Lowercase (span): both characters of İ's lowercase stand for it");
        var bert = JsonNode.Parse("""{"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true, "strip_accents": false, "lowercase": true}""")!.AsObject();
        Check(BpeTokenizer.NormalizerStep(bert)("İ\u200Bx\u0085y\u2028z中") == "i\u0307xy z 中 ", "BertNormalizer: İ lowercased, format and control characters dropped, whitespace a space, spaces around 中");
    }
}
