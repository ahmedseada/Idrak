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

    // What a post-processor does to a single text, laid out once: the offset trims it applies to the text's tokens (ByteLevel's,
    // each with its add_prefix_space, in order), then the parts it joins (the text, sequence 0, with a type id; special
    // tokens with theirs). TokenizerEncodings writes a single text's encoding from it in one pass.
    internal sealed record Layout(bool[] Trims, LayoutPart[] Parts)
    {
        // No post-processor: the text as it is (type id 0).
        public static readonly Layout TextOnly = new([], [LayoutPart.Of(0)]);
    }

    internal readonly record struct LayoutPart(bool Text, int[] SpecialIds, string[]? SpecialTokens, int TypeId)
    {
        public static LayoutPart Of(int typeId) => new(true, [], null, typeId);

        public static LayoutPart Special(int[] ids, string[]? tokens, int typeId) => new(false, ids, tokens, typeId);
    }

    // A library post-processor that can say what it does to a single text (null when its Process would refuse it, so the
    // refusal comes from Process).
    private interface ISingleLayout
    {
        Layout? Single(bool addSpecialTokens);
    }

    // The layout of a single text through post (null: none), or null for a post-processor that cannot give one (an app's).
    internal static Layout? SingleLayout(ITokenizerPostProcessor? post, bool addSpecialTokens) => post switch
    {
        null => Layout.TextOnly,
        ISingleLayout known => known.Single(addSpecialTokens),
        _ => null,
    };

    // The two layouts of a post-processor (with special tokens, without), made when first asked for (null stays a valid
    // answer: Unmade marks one not made yet).
    private sealed class Layouts(Func<bool, Layout?> make)
    {
        private static readonly Layout Unmade = new([], []);
        private Layout? _with = Unmade, _without = Unmade;

        public Layout? Get(bool addSpecialTokens)
        {
            ref Layout? slot = ref addSpecialTokens ? ref _with : ref _without;
            var layout = Volatile.Read(ref slot);
            if (ReferenceEquals(layout, Unmade))
            {
                layout = make(addSpecialTokens);
                Volatile.Write(ref slot, layout);
            }

            return layout;
        }
    }

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
        Dictionary<string, (int[] Ids, string[] Tokens)> specials) : ITokenizerPostProcessor, ISingleLayout
    {
        private Layouts? _layouts;

        public Layout? Single(bool addSpecialTokens) => (_layouts ??= new Layouts(add =>
        {
            var parts = new List<LayoutPart>();
            foreach (var (name, sequence, typeId) in single)
            {
                if (name is not null)
                {
                    if (add)
                    {
                        if (!specials.TryGetValue(name, out var special))
                        {
                            return null;                                      // Process names the missing token
                        }

                        parts.Add(LayoutPart.Special(special.Ids, special.Tokens, typeId));
                    }

                    continue;
                }

                if (sequence != 0)
                {
                    return null;                                              // Process refuses sequence B in the single template
                }

                parts.Add(LayoutPart.Of(typeId));
            }

            return new Layout([], [.. parts]);
        })).Get(addSpecialTokens);

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

        public int AddedTokens(bool isPair)
        {
            if (!isPair && _added >= 0)
            {
                return _added;
            }

            int added = (isPair ? pair : single).Where(p => p.Special is not null).Sum(p => specials[p.Special!].Ids.Length);
            if (!isPair)
            {
                _added = added;
            }

            return added;
        }

        private int _added = -1;                                              // a single text's, counted once (-1: not yet)

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
    private sealed class Bert((string Token, int Id) cls, (string Token, int Id) sep) : ITokenizerPostProcessor, ISingleLayout
    {
        private readonly Layout _with = new([], [LayoutPart.Special([cls.Id], [cls.Token], 0), LayoutPart.Of(0), LayoutPart.Special([sep.Id], [sep.Token], 0)]);

        public int AddedTokens(bool isPair) => isPair ? 3 : 2;

        public Layout? Single(bool addSpecialTokens) => addSpecialTokens ? _with : Layout.TextOnly;

        public TokenizerEncoding Process(TokenizerEncoding encoding, TokenizerEncoding? pair, bool addSpecialTokens) => !addSpecialTokens
            ? Plain(encoding, pair)
            : Join(pair is null
                ? [new Part(null, [cls.Id], 0, -1, [cls.Token]), new Part(encoding, [], 0, 0), new Part(null, [sep.Id], 0, -1, [sep.Token])]
                : [new Part(null, [cls.Id], 0, -1, [cls.Token]), new Part(encoding, [], 0, 0), new Part(null, [sep.Id], 0, -1, [sep.Token]),
                   new Part(pair, [], 1, 1), new Part(null, [sep.Id], 1, -1, [sep.Token])]);
    }

    // RobertaProcessing: offsets trimmed as ByteLevel trims them (trim_offsets), then <s> A </s> (type 0), then </s> B </s>
    // (type 1, as tokenizers sets it).
    private sealed class Roberta((string Token, int Id) cls, (string Token, int Id) sep, bool trimOffsets, bool addPrefixSpace) : ITokenizerPostProcessor, ISingleLayout
    {
        private readonly Layout _with = new(trimOffsets ? [addPrefixSpace] : [],
            [LayoutPart.Special([cls.Id], [cls.Token], 0), LayoutPart.Of(0), LayoutPart.Special([sep.Id], [sep.Token], 0)]);

        private readonly Layout _without = new(trimOffsets ? [addPrefixSpace] : [], [LayoutPart.Of(0)]);

        public int AddedTokens(bool isPair) => isPair ? 4 : 2;

        public Layout? Single(bool addSpecialTokens) => addSpecialTokens ? _with : _without;

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
    private sealed class ByteLevel(bool trimOffsets, bool addPrefixSpace) : ITokenizerPostProcessor, ISingleLayout
    {
        private readonly Layout _layout = new(trimOffsets ? [addPrefixSpace] : [], [LayoutPart.Of(0)]);

        public int AddedTokens(bool isPair) => 0;

        public Layout? Single(bool addSpecialTokens) => _layout;

        // Its trims (none without trim_offsets), as a Sequence applies them first.
        public bool[] Trims => _layout.Trims;

        public TokenizerEncoding Process(TokenizerEncoding encoding, TokenizerEncoding? pair, bool addSpecialTokens) => trimOffsets
            ? Plain(Trim(encoding, addPrefixSpace), pair is null ? null : Trim(pair, addPrefixSpace))
            : Plain(encoding, pair);

        public static TokenizerEncoding Trim(TokenizerEncoding encoding, bool addPrefixSpace)
        {
            if (encoding.Tokens.Length != encoding.Count)
            {
                return encoding;                                              // no tokens to trim by
            }

            var offsets = ((int Start, int End)[])encoding.Offsets.Clone();
            for (int i = 0; i < offsets.Length; i++)
            {
                offsets[i] = Trimmed(encoding.Tokens[i], offsets[i], i, addPrefixSpace);
            }

            return new TokenizerEncoding
            {
                Ids = encoding.Ids, TypeIds = encoding.TypeIds, Offsets = offsets, AttentionMask = encoding.AttentionMask,
                SpecialTokensMask = encoding.SpecialTokensMask, SequenceIds = encoding.SequenceIds, Tokens = encoding.Tokens,
            };
        }
    }

    // One token's offsets trimmed as ByteLevel's trim_offsets trims them (index: the token's place in its text): its leading
    // and trailing spaces (Ġ, or whitespace) cut, but one leading space of a first token when add_prefix_space put it there.
    internal static (int Start, int End) Trimmed(string token, (int Start, int End) offsets, int index, bool addPrefixSpace)
    {
        char space = BpeTokenizer.ByteToChar[' '];
        int leading = 0, trailing = 0;
        while (leading < token.Length && (token[leading] == space || char.IsWhiteSpace(token[leading])))
        {
            leading++;
        }

        while (trailing < token.Length && (token[^(trailing + 1)] == space || char.IsWhiteSpace(token[^(trailing + 1)])))
        {
            trailing++;
        }

        var (start, end) = offsets;
        if (leading > 0)
        {
            bool first = index == 0 || start == 0;
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

        return (start, end);
    }

    // Sequence: the processors that add no tokens first (ByteLevel), then the one that does; tokenizers' files put them so.
    private sealed class Chain(ITokenizerPostProcessor[] processors) : ITokenizerPostProcessor, ISingleLayout
    {
        private Layouts? _layouts;

        public int AddedTokens(bool isPair) => processors.Sum(p => p.AddedTokens(isPair));

        // The ByteLevel trims in order, then the adding processor's layout (its own trims after them).
        public Layout? Single(bool addSpecialTokens) => (_layouts ??= new Layouts(add =>
        {
            var trims = processors.OfType<ByteLevel>().SelectMany(b => b.Trims).ToList();
            var adding = processors.Where(p => p is not ByteLevel).ToArray();
            if (adding.Length > 1)
            {
                return null;                                                  // Process refuses it
            }

            var inner = adding.Length == 0 ? Layout.TextOnly : SingleLayout(adding[0], add);
            return inner is null ? null : new Layout([.. trims, .. inner.Trims], inner.Parts);
        })).Get(addSpecialTokens);

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
