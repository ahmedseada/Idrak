// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.Json.Nodes;
using Idrak.Models.Abstractions;

namespace Idrak.Models;

/// <summary>
/// Reads a BERT-style <c>vocab.txt</c> (one token a line, the line number its id) with its folder's
/// <c>tokenizer_config.json</c> (<c>do_lower_case</c>, <c>strip_accents</c>, <c>tokenize_chinese_chars</c>, the special
/// tokens) into the definition transformers' <c>BertConverter</c> writes as a tokenizer.json, so the two files of one model
/// give the same ids (plan 24, T2).
/// </summary>
internal sealed class VocabTxtReader : ITokenizerFileReader
{
    public string Name => TokenizerFiles.VocabTxt;

    public string Summary => "BERT-style vocab.txt with tokenizer_config.json (WordPiece)";

    public string? Find(string path)
    {
        if (Directory.Exists(path))
        {
            string file = Path.Combine(path, "vocab.txt");
            return File.Exists(file) ? file : null;
        }

        return File.Exists(path) && path.EndsWith(".txt", StringComparison.OrdinalIgnoreCase) ? path : null;
    }

    public TokenizerDefinition Read(string file, TokenizerOptions options)
    {
        // As transformers' load_vocab: each line without its "\n" (a "\r" stays), the line's number its id; a token that
        // repeats takes the later id. The bytes are split here, one string a token.
        byte[] bytes = File.ReadAllBytes(file);
        var vocabulary = new Dictionary<string, int>();
        int id = 0, start = 0;
        var span = bytes.AsSpan();                                            // a byte-order mark stays in the first token, as transformers reads it

        while (start < span.Length)
        {
            int end = span[start..].IndexOf((byte)'\n');
            int length = end < 0 ? span.Length - start : end;
            vocabulary[Encoding.UTF8.GetString(span.Slice(start, length))] = id++;
            start += length + 1;
        }

        string? folder = Path.GetDirectoryName(Path.GetFullPath(file));
        var config = folder is not null && File.Exists(Path.Combine(folder, "tokenizer_config.json"))
            ? JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "tokenizer_config.json"))) as JsonObject ?? []
            : [];
        string Token(string key, string fallback) => config[key] switch
        {
            JsonValue value when value.TryGetValue(out string? text) => text!,
            JsonObject added when added["content"] is JsonValue content => (string)content!,    // AddedToken written out
            _ => fallback,
        };

        return BertDefinition(file, vocabulary, (bool?)config["do_lower_case"] ?? true, config["strip_accents"]?.DeepClone(),
            (bool?)config["tokenize_chinese_chars"] ?? true, Token("unk_token", "[UNK]"), Token("sep_token", "[SEP]"),
            Token("cls_token", "[CLS]"), Token("pad_token", "[PAD]"), Token("mask_token", "[MASK]"));
    }

    // A BERT tokenizer as transformers' BertConverter writes it, for a vocabulary of token to id (vocab.txt here, a GGUF
    // "bert" tokenizer too): BertNormalizer, BertPreTokenizer, [CLS] A [SEP] (B [SEP]), the WordPiece decoder and model,
    // and the special tokens the vocabulary holds as added tokens.
    internal static TokenizerDefinition BertDefinition(string source, Dictionary<string, int> vocabulary, bool lowercase, JsonNode? stripAccents,
        bool chinese, string unknown, string sep, string cls, string pad, string mask)
    {
        int IdOf(string token) => vocabulary.TryGetValue(token, out int found) ? found
            : throw new InvalidDataException($"{source}: the special token '{token}' is not in the vocabulary.");

        var added = new JsonArray();
        foreach (var (token, tokenId) in new[] { pad, unknown, cls, sep, mask }.Distinct(StringComparer.Ordinal).Where(vocabulary.ContainsKey)
                     .Select(t => (t, vocabulary[t])).OrderBy(t => t.Item2))
        {
            added.Add((JsonNode)new JsonObject
            {
                ["id"] = tokenId, ["content"] = token, ["single_word"] = false, ["lstrip"] = false, ["rstrip"] = false,
                ["normalized"] = false, ["special"] = true,
            });
        }

        JsonObject Special(string name, int type = 0) => new() { ["SpecialToken"] = new JsonObject { ["id"] = name, ["type_id"] = type } };
        JsonObject Sequence(string name, int type) => new() { ["Sequence"] = new JsonObject { ["id"] = name, ["type_id"] = type } };
        var json = new JsonObject
        {
            ["added_tokens"] = added,
            ["normalizer"] = new JsonObject
            {
                ["type"] = "BertNormalizer", ["clean_text"] = true, ["handle_chinese_chars"] = chinese, ["strip_accents"] = stripAccents,
                ["lowercase"] = lowercase,
            },
            ["pre_tokenizer"] = new JsonObject { ["type"] = "BertPreTokenizer" },
            ["post_processor"] = new JsonObject
            {
                ["type"] = "TemplateProcessing",
                ["single"] = new JsonArray(Special(cls), Sequence("A", 0), Special(sep)),
                ["pair"] = new JsonArray(Special(cls), Sequence("A", 0), Special(sep), Sequence("B", 1), Special(sep, 1)),
                ["special_tokens"] = new JsonObject
                {
                    [cls] = new JsonObject { ["id"] = cls, ["ids"] = new JsonArray(IdOf(cls)), ["tokens"] = new JsonArray(cls) },
                    [sep] = new JsonObject { ["id"] = sep, ["ids"] = new JsonArray(IdOf(sep)), ["tokens"] = new JsonArray(sep) },
                },
            },
            ["decoder"] = new JsonObject { ["type"] = "WordPiece", ["prefix"] = "##", ["cleanup"] = true },
            ["model"] = new JsonObject
            {
                ["type"] = "WordPiece", ["unk_token"] = unknown, ["continuing_subword_prefix"] = "##", ["max_input_chars_per_word"] = 100,
            },
        };
        return new TokenizerDefinition { Source = source, Json = json, Vocabulary = vocabulary };
    }
}
