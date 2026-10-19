// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Models.Abstractions;

namespace Idrak.Models;

/// <summary>
/// The built-in tokenizer.json post-processors (plan 24, T2), as Hugging Face's <c>tokenizers</c> defines them:
/// <c>TemplateProcessing</c> (special tokens and type ids by a template), <c>BertProcessing</c> (<c>[CLS] A [SEP] B [SEP]</c>),
/// <c>RobertaProcessing</c> (<c>&lt;s&gt; A &lt;/s&gt;&lt;/s&gt; B &lt;/s&gt;</c>), <c>ByteLevel</c> (no tokens; it trims
/// offsets) and <c>Sequence</c>. Registered in <see cref="TokenizerComponents"/>, so an app's type overrides one.
/// </summary>
internal static class PostProcessors
{
    // The built-in types, as TokenizerComponents lists them.
    public static readonly string[] Types = ["TemplateProcessing", "BertProcessing", "RobertaProcessing", "ByteLevel", "Sequence"];

    // The post-processor a tokenizer.json names (an app's registered type first), or null for none.
    public static ITokenizerPostProcessor? Of(JsonObject? spec)
    {
        if (spec is null)
        {
            return null;
        }

        return TokenizerComponents.FindPostProcessor((string?)spec["type"]) is { } create ? create(spec) : BuiltIn(spec);
    }

    // The library's post-processor of the spec's type.
    public static ITokenizerPostProcessor BuiltIn(JsonObject spec)
    {
        string type = (string)spec["type"]!;
        return type switch
        {
            "TemplateProcessing" => Template.Read(spec),
            "BertProcessing" => new Bert(Special(spec["cls"]), Special(spec["sep"])),
            "RobertaProcessing" => new Roberta(Special(spec["cls"]), Special(spec["sep"]), (bool?)spec["trim_offsets"] ?? true, (bool?)spec["add_prefix_space"] ?? true),
            "ByteLevel" => new ByteLevel((bool?)spec["trim_offsets"] ?? true, (bool?)spec["add_prefix_space"] ?? true),
            "Sequence" => new Chain([.. spec["processors"]!.AsArray().OfType<JsonObject>().Select(p => Of(p)!)]),
            _ => throw new NotSupportedException($"Tokenizer post-processor '{type}' is not supported; add it with TokenizerComponents.RegisterPostProcessor."),
        };
    }

    // ["[CLS]", 101] as a special token.
    private static (string Token, int Id) Special(JsonNode? pair) =>
        pair is JsonArray { Count: 2 } a ? ((string)a[0]!, (int)a[1]!) : throw new InvalidDataException($"A post-processor's special token is not [token, id]: {pair?.ToJsonString()}.");

    // A run of tokens to join: from an encoding (with its type id and sequence id) or special ones (with their tokens).
    private readonly record struct Part(TokenizerEncoding? Encoding, int[] SpecialIds, int TypeId, int Sequence, string[]? SpecialTokens = null);

    // The parts joined into one encoding.
    private static TokenizerEncoding Join(IReadOnlyList<Part> parts)
    {
        int count = parts.Sum(p => p.Encoding?.Count ?? p.SpecialIds.Length);
        var ids = new int[count];
        var types = new int[count];
        var offsets = new (int Start, int End)[count];
        var attention = new int[count];
        var special = new int[count];
        var sequences = new int[count];
        bool named = parts.All(p => p.Encoding is { } e ? e.Tokens.Length == e.Count : p.SpecialTokens?.Length == p.SpecialIds.Length);
        var tokens = named ? new string[count] : [];
        int at = 0;
        foreach (var part in parts)
        {
            if (part.Encoding is { } e)
            {
                if (named)
                {
                    e.Tokens.CopyTo(tokens, at);
                }

                e.Ids.CopyTo(ids, at);
                Array.Fill(types, part.TypeId, at, e.Count);
                e.Offsets.CopyTo(offsets, at);
                e.AttentionMask.CopyTo(attention, at);
                e.SpecialTokensMask.CopyTo(special, at);
                Array.Fill(sequences, part.Sequence, at, e.Count);
                at += e.Count;
                continue;
            }

            for (int k = 0; k < part.SpecialIds.Length; k++)
            {
                (ids[at], types[at], offsets[at], attention[at], special[at], sequences[at]) = (part.SpecialIds[k], part.TypeId, (0, 0), 1, 1, -1);
                if (named)
                {
                    tokens[at] = part.SpecialTokens![k];
                }

                at++;
            }
        }

        return new TokenizerEncoding { Ids = ids, TypeIds = types, Offsets = offsets, AttentionMask = attention, SpecialTokensMask = special, SequenceIds = sequences, Tokens = tokens };
    }

    // Without special tokens: the texts joined as they are, each its sequence (tokenizers' default_process), type ids kept.
    internal static TokenizerEncoding Plain(TokenizerEncoding encoding, TokenizerEncoding? pair)
    {
        if (pair is null)
        {
            return encoding;
        }

        var joined = Join([new Part(encoding, [], 0, 0), new Part(pair, [], 0, 1)]);
        encoding.TypeIds.CopyTo(joined.TypeIds, 0);
        pair.TypeIds.CopyTo(joined.TypeIds, encoding.Count);
        return joined;
    }

    // TemplateProcessing: the single or pair template's pieces in order; special tokens only when asked for, but the
    // sequences' type ids always as the template sets them.
    private sealed class Template(List<(string? Special, int Sequence, int TypeId)> single, List<(string? Special, int Sequence, int TypeId)> pair,
        Dictionary<string, (int[] Ids, string[] Tokens)> specials) : ITokenizerPostProcessor
    {
        public static Template Read(JsonObject spec)
        {
            var specials = new Dictionary<string, (int[] Ids, string[] Tokens)>(StringComparer.Ordinal);
            foreach (var (name, value) in spec["special_tokens"]?.AsObject() ?? [])
            {
                specials[name] = ([.. value!["ids"]!.AsArray().Select(i => (int)i!)], [.. (value["tokens"]?.AsArray() ?? []).Select(t => (string)t!)]);
            }

            return new Template(Pieces(spec["single"]), Pieces(spec["pair"]), specials);
        }

        private static List<(string? Special, int Sequence, int TypeId)> Pieces(JsonNode? node) =>
            [.. (node?.AsArray() ?? []).Select(p => p!["SpecialToken"] is JsonObject s
                ? ((string?)(string)s["id"]!, -1, (int?)s["type_id"] ?? 0)
                : ((string?)null, (string)p["Sequence"]!["id"]! == "A" ? 0 : 1, (int?)p["Sequence"]!["type_id"] ?? 0))];

        public int AddedTokens(bool isPair) => (isPair ? pair : single).Where(p => p.Special is not null).Sum(p => specials[p.Special!].Ids.Length);

        public TokenizerEncoding Process(TokenizerEncoding encoding, TokenizerEncoding? second, bool addSpecialTokens)
        {
            var parts = new List<Part>();
            foreach (var (name, sequence, typeId) in second is null ? single : pair)
            {
                if (name is not null)
                {
                    if (addSpecialTokens)
                    {
                        var (ids, tokens) = specials.TryGetValue(name, out var special) ? special
                            : throw new InvalidDataException($"TemplateProcessing names the special token '{name}' but does not define it.");
                        parts.Add(new Part(null, ids, typeId, -1, tokens));
                    }

                    continue;
                }

                var text = sequence == 0 ? encoding : second ?? throw new InvalidOperationException("The single template names sequence B.");
                parts.Add(new Part(text, [], typeId, sequence));
            }

            return Join(parts);
        }
    }

    // BertProcessing: [CLS] A [SEP] (type 0), then B [SEP] (type 1).
    private sealed class Bert((string Token, int Id) cls, (string Token, int Id) sep) : ITokenizerPostProcessor
    {
        public int AddedTokens(bool isPair) => isPair ? 3 : 2;

        public TokenizerEncoding Process(TokenizerEncoding encoding, TokenizerEncoding? pair, bool addSpecialTokens) => !addSpecialTokens
            ? Plain(encoding, pair)
            : Join(pair is null
                ? [new Part(null, [cls.Id], 0, -1, [cls.Token]), new Part(encoding, [], 0, 0), new Part(null, [sep.Id], 0, -1, [sep.Token])]
                : [new Part(null, [cls.Id], 0, -1, [cls.Token]), new Part(encoding, [], 0, 0), new Part(null, [sep.Id], 0, -1, [sep.Token]),
                   new Part(pair, [], 1, 1), new Part(null, [sep.Id], 1, -1, [sep.Token])]);
    }

    // RobertaProcessing: offsets trimmed as ByteLevel trims them (trim_offsets), then <s> A </s> (type 0), then </s> B </s>
    // (type 1, as tokenizers sets it).
    private sealed class Roberta((string Token, int Id) cls, (string Token, int Id) sep, bool trimOffsets, bool addPrefixSpace) : ITokenizerPostProcessor
    {
        public int AddedTokens(bool isPair) => isPair ? 4 : 2;

        public TokenizerEncoding Process(TokenizerEncoding encoding, TokenizerEncoding? pair, bool addSpecialTokens)
        {
            if (trimOffsets)
            {
                encoding = ByteLevel.Trim(encoding, addPrefixSpace);
                pair = pair is null ? null : ByteLevel.Trim(pair, addPrefixSpace);
            }

            return !addSpecialTokens
                ? Plain(encoding, pair)
                : Join(pair is null
                    ? [new Part(null, [cls.Id], 0, -1, [cls.Token]), new Part(encoding, [], 0, 0), new Part(null, [sep.Id], 0, -1, [sep.Token])]
                    : [new Part(null, [cls.Id], 0, -1, [cls.Token]), new Part(encoding, [], 0, 0), new Part(null, [sep.Id], 0, -1, [sep.Token]),
                       new Part(null, [sep.Id], 1, -1, [sep.Token]), new Part(pair, [], 1, 1), new Part(null, [sep.Id], 1, -1, [sep.Token])]);
        }
    }

    // ByteLevel as a post-processor adds no tokens; with trim_offsets, a token's leading and trailing spaces (Ġ, or
    // whitespace) are cut from its offsets, as tokenizers' process_offsets does (one leading space of a first token stays
    // when add_prefix_space put it there).
    private sealed class ByteLevel(bool trimOffsets, bool addPrefixSpace) : ITokenizerPostProcessor
    {
        public int AddedTokens(bool isPair) => 0;

        public TokenizerEncoding Process(TokenizerEncoding encoding, TokenizerEncoding? pair, bool addSpecialTokens) => trimOffsets
            ? Plain(Trim(encoding, addPrefixSpace), pair is null ? null : Trim(pair, addPrefixSpace))
            : Plain(encoding, pair);

        public static TokenizerEncoding Trim(TokenizerEncoding encoding, bool addPrefixSpace)
        {
            if (encoding.Tokens.Length != encoding.Count)
            {
                return encoding;                                              // no tokens to trim by
            }

            char space = BpeTokenizer.ByteToChar[' '];
            var offsets = ((int Start, int End)[])encoding.Offsets.Clone();
            for (int i = 0; i < offsets.Length; i++)
            {
                string token = encoding.Tokens[i];
                int leading = 0, trailing = 0;
                while (leading < token.Length && (token[leading] == space || char.IsWhiteSpace(token[leading])))
                {
                    leading++;
                }

                while (trailing < token.Length && (token[^(trailing + 1)] == space || char.IsWhiteSpace(token[^(trailing + 1)])))
                {
                    trailing++;
                }

                var (start, end) = offsets[i];
                if (leading > 0)
                {
                    bool first = i == 0 || start == 0;
                    if (first && addPrefixSpace && leading == 1)
                    {
                        leading = 0;
                    }

                    start = Math.Min(start + leading, end);
                }

                if (trailing > 0 && end >= trailing)
                {
                    end = Math.Max(end - trailing, start);
                }

                offsets[i] = (start, end);
            }

            return new TokenizerEncoding
            {
                Ids = encoding.Ids, TypeIds = encoding.TypeIds, Offsets = offsets, AttentionMask = encoding.AttentionMask,
                SpecialTokensMask = encoding.SpecialTokensMask, SequenceIds = encoding.SequenceIds, Tokens = encoding.Tokens,
            };
        }
    }

    // Sequence: the processors that add no tokens first (ByteLevel), then the one that does; tokenizers' files put them so.
    private sealed class Chain(ITokenizerPostProcessor[] processors) : ITokenizerPostProcessor
    {
        public int AddedTokens(bool isPair) => processors.Sum(p => p.AddedTokens(isPair));

        public TokenizerEncoding Process(TokenizerEncoding encoding, TokenizerEncoding? pair, bool addSpecialTokens)
        {
            // The offset-only processors (ByteLevel) on each text first, then the one that adds tokens.
            foreach (var trimming in processors.OfType<ByteLevel>())
            {
                encoding = trimming.Process(encoding, null, false);
                pair = pair is null ? null : trimming.Process(pair, null, false);
            }

            var adding = processors.Where(p => p is not ByteLevel).ToArray();
            if (adding.Length > 1)
            {
                throw new NotSupportedException("A Sequence post-processor with more than one that adds tokens is not supported.");
            }

            return adding.Length == 0 ? Plain(encoding, pair) : adding[0].Process(encoding, pair, addSpecialTokens);
        }
    }
}
