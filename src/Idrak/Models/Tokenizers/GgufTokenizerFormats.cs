// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Models.Abstractions;

namespace Idrak.Models;

/// <summary>
/// The built-in GGUF tokenizer converters (<see cref="GgufTokenizers"/>; plan 24, T3), by <c>tokenizer.ggml.model</c>:
/// "gpt2" (byte-level BPE), "llama" and "t5" (a SentencePiece BPE or Unigram model: the GGUF's fields are written back as
/// the SentencePiece ModelProto they came from and read as a tokenizer.model is, so both files of a model give the same
/// tokenizer), and "bert" (WordPiece: llama.cpp stores word-initial pieces with ▁ and continuations bare, turned back here).
/// </summary>
internal static class GgufTokenizerFormats
{
    // One of the built-in converters, registered in GgufTokenizers under its tokenizer model's name.
    internal sealed class Converter(string name, string summary, Func<GgufFile, IReadOnlyList<string>, List<string>, TokenizerDefinition> convert) : IGgufTokenizer
    {
        public string Name => name;

        public string Summary => summary;

        public IReadOnlySet<string> Keys => OptionKeys.None;

        public TokenizerDefinition Create(GgufTokenizerRequest request) => convert(request.File, request.Tokens, request.Notes);
    }

    // Byte-level BPE: the vocabulary and merges, token types 3 and 4 as added tokens (3 special), the pre-tokenizer's split
    // pattern by tokenizer.ggml.pre (GgufPreTokenizers, or its fallback for a name nobody registered), the ByteLevel decoder.
    public static TokenizerDefinition Gpt2(GgufFile file, IReadOnlyList<string> tokens, List<string> notes)
    {
        var types = file.Get<long[]>("tokenizer.ggml.token_type", []);
        var merges = file.Get<string[]>("tokenizer.ggml.merges", []);
        var vocabulary = new Dictionary<string, int>(tokens.Count);
        var added = new JsonArray();
        for (int id = 0; id < tokens.Count; id++)
        {
            long type = id < types.Length ? types[id] : 1;
            if (type is 3 or 4)
            {
                added.Add((JsonNode)new JsonObject { ["id"] = id, ["content"] = tokens[id], ["special"] = type == 3 });
            }

            vocabulary.TryAdd(tokens[id], id);                                 // a repeated token keeps its first id
        }

        string pre = file.Get("tokenizer.ggml.pre", "default");
        if (!GgufPreTokenizers.TryGet(pre, out string? pattern))
        {
            if (GgufPreTokenizers.Fallback is not { } fallback || !GgufPreTokenizers.TryGet(fallback, out pattern))
            {
                throw new NotRegisteredException(nameof(GgufPreTokenizers), pre, GgufPreTokenizers.Names,
                    $"{file.Path}: GGUF pre-tokenizer '{pre}' is not registered (registered: {string.Join(", ", GgufPreTokenizers.Names)}), "
                    + "and no registered fallback is set (GgufPreTokenizers.Fallback); register the family's pattern with GgufPreTokenizers.Register, "
                    + "or reference Idrak.Text.Plugins and call TextPlugins.RegisterAll().");
            }

            notes.Add($"pre-tokenizer '{pre}' is not registered; the splitting rule of '{fallback}' (GgufPreTokenizers.Fallback) is used (tokens may differ slightly from the original; " +
                "register the family's pattern with GgufPreTokenizers.Register).");
        }

        var pretokenizers = new JsonArray();
        if (pattern is not null)
        {
            pretokenizers.Add((JsonNode)new JsonObject { ["type"] = "Split", ["pattern"] = new JsonObject { ["Regex"] = pattern }, ["behavior"] = "Isolated", ["invert"] = false });
        }

        pretokenizers.Add((JsonNode)new JsonObject { ["type"] = "ByteLevel", ["add_prefix_space"] = false, ["trim_offsets"] = false, ["use_regex"] = pattern is null });
        var json = new JsonObject
        {
            ["added_tokens"] = added,
            ["normalizer"] = null,
            ["pre_tokenizer"] = new JsonObject { ["type"] = "Sequence", ["pretokenizers"] = pretokenizers },
            ["decoder"] = new JsonObject { ["type"] = "ByteLevel" },
            ["model"] = new JsonObject { ["type"] = TokenizerModels.Bpe },
        };
        return new TokenizerDefinition { Source = file.Path, Json = json, Vocabulary = vocabulary, Merges = [.. merges.Select(BpeTokenizer.Split)] };
    }

    // SentencePiece BPE (llama.cpp's SPM: Llama 2, Mistral 7B, TinyLlama): no whitespace collapsing unless the file says so.
    public static TokenizerDefinition Llama(GgufFile file, IReadOnlyList<string> tokens, List<string> notes) =>
        SentencePiece(file, tokens, notes, SentencePieceAlgorithm.Bpe, removeExtraWhitespaces: false);

    // SentencePiece Unigram (llama.cpp's UGM: T5), with the precompiled character map the file carries.
    public static TokenizerDefinition T5(GgufFile file, IReadOnlyList<string> tokens, List<string> notes) =>
        SentencePiece(file, tokens, notes, SentencePieceAlgorithm.Unigram, removeExtraWhitespaces: true);

    private static TokenizerDefinition SentencePiece(GgufFile file, IReadOnlyList<string> tokens, List<string> notes, SentencePieceAlgorithm algorithm,
        bool removeExtraWhitespaces)
    {
        float[] scores = file.Metadata.TryGetValue("tokenizer.ggml.scores", out var raw) ? raw switch
        {
            float[] floats => floats,
            double[] doubles => [.. doubles.Select(d => (float)d)],
            _ => throw new InvalidDataException($"{file.Path}: tokenizer.ggml.scores is not an array of numbers."),
        } : new float[tokens.Count];
        if (raw is null)
        {
            notes.Add("the file has no token scores (tokenizer.ggml.scores): every piece scores 0, so pieces may split differently from the original.");
        }

        var typeValues = file.Get<long[]>("tokenizer.ggml.token_type", []);
        var types = new SentencePieceType[tokens.Count];
        for (int id = 0; id < types.Length; id++)
        {
            long type = id < typeValues.Length ? typeValues[id] : 1;
            types[id] = type is >= 1 and <= 6 ? (SentencePieceType)type : SentencePieceType.Normal;
        }

        int unknownPiece = Array.IndexOf(types, SentencePieceType.Unknown);
        var settings = new SentencePieceWriter.Settings
        {
            Algorithm = algorithm,
            ByteFallback = types.Contains(SentencePieceType.Byte),
            UnkId = Id(file, "tokenizer.ggml.unknown_token_id", unknownPiece >= 0 ? unknownPiece : 0),
            BosId = Id(file, "tokenizer.ggml.bos_token_id", -1),
            EosId = Id(file, "tokenizer.ggml.eos_token_id", -1),
            PadId = Id(file, "tokenizer.ggml.padding_token_id", -1),
            PrecompiledCharsmap = file.Metadata.ContainsKey("tokenizer.ggml.precompiled_charsmap") ? file.Get<byte[]>("tokenizer.ggml.precompiled_charsmap", []) : [],
            AddDummyPrefix = file.Get("tokenizer.ggml.add_space_prefix", true),
            RemoveExtraWhitespaces = file.Get("tokenizer.ggml.remove_extra_whitespaces", removeExtraWhitespaces),
        };
        settings = settings with { NormalizerName = settings.PrecompiledCharsmap.Length > 0 ? "precompiled" : "identity" };
        var model = SentencePieceModel.Read(SentencePieceWriter.Write(tokens, scores, types, settings), file.Path);
        return SentencePieceReader.FromModel(model);
    }

    // WordPiece: llama.cpp's converter writes "[X]" as it is, a continuation "##x" as "x" and any other piece "x" as "▁x";
    // turned back, with the special tokens by their ids. GGUF does not say whether the model lowercases: llama.cpp's
    // tokenizer always does, and so does this one.
    public static TokenizerDefinition Bert(GgufFile file, IReadOnlyList<string> tokens, List<string> notes)
    {
        var vocabulary = new Dictionary<string, int>(tokens.Count);
        for (int id = 0; id < tokens.Count; id++)
        {
            string token = tokens[id];
            string original = token.Length >= 2 && token[0] == '[' && token[^1] == ']' ? token
                : token.StartsWith('▁') ? token[1..]
                : "##" + token;
            vocabulary.TryAdd(original, id);
        }

        string Token(string key, string fallback) => Id(file, key, -1) is var id && id >= 0 && id < tokens.Count ? tokens[id] : fallback;
        notes.Add("a GGUF BERT tokenizer does not say whether it lowercases; it lowercases and strips accents, as llama.cpp's does.");
        return VocabTxtReader.BertDefinition(file.Path, vocabulary, lowercase: true, stripAccents: null, chinese: true,
            Token("tokenizer.ggml.unknown_token_id", "[UNK]"), Token("tokenizer.ggml.seperator_token_id", "[SEP]"),
            Token("tokenizer.ggml.cls_token_id", "[CLS]"), Token("tokenizer.ggml.padding_token_id", "[PAD]"), Token("tokenizer.ggml.mask_token_id", "[MASK]"));
    }

    // An integer metadata value (llama.cpp writes ids as uint32), or the fallback when the file has none.
    private static int Id(GgufFile file, string key, int fallback) =>
        file.Metadata.TryGetValue(key, out var value) && value is not Array ? (int)Convert.ToInt64(value, CultureInfo.InvariantCulture) : fallback;
}
