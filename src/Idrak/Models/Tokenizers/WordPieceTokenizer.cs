// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Collections.Concurrent;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Idrak.Models.Abstractions;

namespace Idrak.Models;

/// <summary>
/// The WordPiece model (BERT and most embedding and classification models; plan 24, T2), as Hugging Face's
/// <c>tokenizers</c> runs it: added tokens matched first, the tokenizer.json's normalizer and pre-tokenizer on spans
/// (<see cref="TokenizerPipeline"/>), then each word split greedily into the longest pieces of the vocabulary, the first
/// from the word's start, the rest continuation pieces (<c>##</c>); a word with no such split, or longer than
/// <c>max_input_chars_per_word</c>, is one unknown token. <see cref="Encode(string)"/> adds no special tokens (the post-processor
/// does, when asked). Built by the "WordPiece" model of <see cref="TokenizerModels"/>.
/// </summary>
internal sealed class WordPieceTokenizer : IPretrainedTokenizer
{
    private const int StackBytes = 256;                                       // words up to this many UTF-8 bytes stay on the stack
    private const int StackMatches = 32;

    private readonly JsonObject _source;                                      // the definition's JSON (no vocabulary), for Save
    private readonly Dictionary<string, int> _vocabulary;                     // model entries and added tokens
    private readonly Dictionary<string, int>.AlternateLookup<ReadOnlySpan<char>> _vocabularySpans;
    private readonly HashSet<string> _outsideModel;                           // added tokens the model's vocabulary lacks
    private string[] _tokens;
    private readonly DoubleArrayTrie _words;                                  // entries a word may start with
    private readonly DoubleArrayTrie? _continuations;                         // continuation entries, without their prefix
    private readonly int _unknown;
    private readonly int _maxChars;
    private readonly List<(string Content, int Id)> _added;
    private readonly HashSet<int> _special;
    private readonly Regex? _addedPattern;
    private readonly TokenizerPipeline _pipeline;
    private readonly Func<List<string>, List<string>>? _decoder;
    private readonly int _cacheLimit;
    private readonly int _workers;
    private TokenizerEncodings.RawEncoder? _raw;                             // Raw, as Finish takes it (made once)
    private readonly ConcurrentDictionary<string, int[]> _cache = new();
    private readonly ConcurrentDictionary<string, int[]>.AlternateLookup<ReadOnlySpan<char>> _cacheSpans;

    [ThreadStatic] private static PreTokenizedText? t_pieces;                 // one per thread, its buffers kept between calls

    private WordPieceTokenizer(TokenizerDefinition definition, TokenizerOptions options)
    {
        _source = definition.Json;
        var model = definition.Json["model"]!.AsObject();
        var vocabulary = definition.Vocabulary
            ?? throw new InvalidDataException($"{definition.Source}: the WordPiece model has no vocabulary of token to id (model.vocab).");
        string unknown = (string?)model["unk_token"] ?? "[UNK]";
        string prefix = (string?)model["continuing_subword_prefix"] ?? "##";
        _maxChars = options.ResolvedMaxInputCharsPerWord((int?)model["max_input_chars_per_word"] ?? 100);
        _unknown = vocabulary.TryGetValue(unknown, out int u) ? u
            : throw new InvalidDataException($"{definition.Source}: the WordPiece unknown token '{unknown}' is not in the vocabulary.");

        // The tries: every entry a word may start with, and the continuation entries without their prefix (so no
        // "##" + piece is built to look one up). Keys a text cannot hold (empty, U+0000) are left out.
        var words = new List<string>(vocabulary.Count);
        var wordIds = new List<int>(vocabulary.Count);
        var continuations = new List<string>();
        var continuationIds = new List<int>();
        foreach (var (token, id) in vocabulary)
        {
            if (token.Length == 0 || token.Contains('\0', StringComparison.Ordinal))
            {
                continue;
            }

            words.Add(token);
            wordIds.Add(id);
            if (prefix.Length > 0 && token.Length > prefix.Length && token.StartsWith(prefix, StringComparison.Ordinal))
            {
                continuations.Add(token[prefix.Length..]);
                continuationIds.Add(id);
            }
        }

        _words = DoubleArrayTrie.Build(words, wordIds);
        _continuations = prefix.Length == 0 ? _words : continuations.Count > 0 ? DoubleArrayTrie.Build(continuations, continuationIds) : null;

        _vocabulary = vocabulary;
        var addedTokens = definition.Json["added_tokens"]?.AsArray() ?? [];
        _added = [.. addedTokens.Select(t => ((string)t!["content"]!, (int)t["id"]!))];
        _special = [.. addedTokens.Where(t => (bool?)t!["special"] ?? false).Select(t => (int)t!["id"]!)];
        _outsideModel = [.. _added.Where(a => !_vocabulary.ContainsKey(a.Content)).Select(a => a.Content)];
        foreach (var (content, id) in _added)
        {
            _vocabulary[content] = id;
        }

        _vocabularySpans = _vocabulary.GetAlternateLookup<ReadOnlySpan<char>>();

        int size = 0;
        foreach (int id in _vocabulary.Values)
        {
            size = Math.Max(size, id + 1);
        }

        _tokens = new string[size];
        foreach (var (token, id) in _vocabulary)
        {
            _tokens[id] = token;
        }

        if (_added.Count > 0)
        {
            _addedPattern = new Regex(string.Join('|', _added.Select(a => a.Content).OrderByDescending(c => c.Length).Select(Regex.Escape)), BpeTokenizer.Fast);
        }

        _pipeline = new TokenizerPipeline(definition.Json);
        _decoder = definition.Json["decoder"] is JsonObject decoder ? Decoder(decoder) : null;
        PostProcessor = PostProcessors.Of(definition.Json["post_processor"] as JsonObject);
        _cacheLimit = options.ResolvedPieceCacheLimit();
        _workers = options.ResolvedWorkers();
        _cacheSpans = _cache.GetAlternateLookup<ReadOnlySpan<char>>();
    }

    // The decoder a tokenizer.json names, made once: an app's registered type, else the built-in stages.
    private static Func<List<string>, List<string>> Decoder(JsonObject spec)
    {
        if (TokenizerComponents.FindDecoder((string?)spec["type"]) is not { } create)
        {
            return BpeTokenizer.DecoderStage(spec);
        }

        var component = create(spec);
        return tokens => component.Decode(tokens) is var result && result is List<string> list ? list : [.. result];
    }

    // The library's "WordPiece" model (TokenizerModels): the tokenizer of a definition, which it takes as its own.
    internal static WordPieceTokenizer FromDefinition(TokenizerDefinition definition, TokenizerOptions options) => new(definition, options);

    /// <summary>The post-processor the tokenizer.json names (special tokens around a text or a pair), or null.</summary>
    internal ITokenizerPostProcessor? PostProcessor { get; }

    public string Model => TokenizerModels.WordPiece;

    public int VocabularySize => _tokens.Length;

    public int? IdOf(string token) => _vocabulary.TryGetValue(token, out int id) ? id : null;

    public string TokenOf(int id) => (uint)id < (uint)_tokens.Length ? _tokens[id] ?? "" : "";

    public bool IsSpecial(int id) => _special.Contains(id);

    public void PadVocabulary(int size)
    {
        if (size > _tokens.Length)
        {
            Array.Resize(ref _tokens, size);
        }
    }

    public IReadOnlyList<int> Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        text = BpeTokenizer.WithoutLoneSurrogates(text);
        var ids = new List<int>(text.Length / 3 + 4);
        EncodeSegments(text, 0, text.Length, ids);
        return ids;
    }

    public void EncodeRange(string text, int start, int length, List<int> ids)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(ids);
        if (text.AsSpan(start, length).ContainsAnyInRange('\uD800', '\uDFFF'))
        {
            text = BpeTokenizer.WithoutLoneSurrogates(text.Substring(start, length));   // the range may cut a pair in half
            start = 0;
        }

        EncodeSegments(text, start, length, ids);
    }

    // Added tokens, and the text between them, of text[start..start + length].
    private void EncodeSegments(string text, int start, int length, List<int> ids)
    {
        var pieces = t_pieces ?? new PreTokenizedText();
        t_pieces = null;                                                      // taken (a nested call makes its own)
        try
        {
            int last = start;
            if (_addedPattern is not null)
            {
                foreach (var match in _addedPattern.EnumerateMatches(text.AsSpan(start, length)))
                {
                    int at = start + match.Index;
                    EncodeText(text, last, at - last, last == start, ids, pieces);
                    ids.Add(_vocabularySpans[text.AsSpan(at, match.Length)]);
                    last = at + match.Length;
                }
            }

            EncodeText(text, last, start + length - last, last == start, ids, pieces);
        }
        finally
        {
            t_pieces = pieces;
        }
    }

    private void EncodeText(string text, int start, int length, bool atStart, List<int> ids, PreTokenizedText pieces)
    {
        if (length == 0)
        {
            return;
        }

        _pipeline.Run(text, start, length, atStart, pieces);
        for (int i = 0; i < pieces.Pieces.Count; i++)
        {
            EncodeWord(pieces.Piece(i), ids);
        }
    }

    // One word: its cached ids, else the greedy longest-match split (or the unknown token), cached.
    private void EncodeWord(ReadOnlySpan<char> word, List<int> ids)
    {
        if (word.Length == 0)
        {
            return;
        }

        if (_cacheSpans.TryGetValue(word, out var cached))
        {
            ids.AddRange(cached);
            return;
        }

        int before = ids.Count;
        Split(word, ids, null);
        if (_cache.Count < _cacheLimit)
        {
            _cache.TryAdd(new string(word), System.Runtime.InteropServices.CollectionsMarshal.AsSpan(ids)[before..].ToArray());
        }
    }

    // The greedy split of a word; ranges (when given) gets each token's characters in the word.
    private void Split(ReadOnlySpan<char> word, List<int> ids, List<(int Start, int Length)>? ranges)
    {
        int characters = 0;
        for (int i = 0; i < word.Length; i++)
        {
            if (!char.IsLowSurrogate(word[i]))
            {
                characters++;
            }
        }

        if (characters > _maxChars)
        {
            ids.Add(_unknown);
            ranges?.Add((0, word.Length));
            return;
        }

        int count = Encoding.UTF8.GetByteCount(word);
        byte[]? rentedBytes = null;
        TrieMatch[]? rentedMatches = null;
        Span<byte> bytes = count <= StackBytes ? stackalloc byte[StackBytes] : (rentedBytes = ArrayPool<byte>.Shared.Rent(count));
        Span<TrieMatch> matches = stackalloc TrieMatch[StackMatches];
        int before = ids.Count, rangesBefore = ranges?.Count ?? 0, character = 0;
        try
        {
            Encoding.UTF8.GetBytes(word, bytes);
            for (int at = 0; at < count;)
            {
                var trie = at == 0 ? _words : _continuations;
                int found = trie?.CommonPrefixSearch(bytes[at..count], matches) ?? 0;
                if (found == 0)
                {
                    ids.RemoveRange(before, ids.Count - before);              // no split: the word is unknown
                    ids.Add(_unknown);
                    ranges?.RemoveRange(rangesBefore, ranges.Count - rangesBefore);
                    ranges?.Add((0, word.Length));
                    return;
                }

                TrieMatch longest;
                if (found <= matches.Length)
                {
                    longest = matches[found - 1];
                }
                else
                {
                    if (rentedMatches is null || rentedMatches.Length < found)
                    {
                        if (rentedMatches is not null)
                        {
                            ArrayPool<TrieMatch>.Shared.Return(rentedMatches);
                        }

                        rentedMatches = ArrayPool<TrieMatch>.Shared.Rent(found);
                    }

                    trie!.CommonPrefixSearch(bytes[at..count], rentedMatches);
                    longest = rentedMatches[found - 1];
                }

                ids.Add(longest.Value);
                if (ranges is not null)
                {
                    int chars = Encoding.UTF8.GetCharCount(bytes.Slice(at, longest.Length));
                    ranges.Add((character, chars));
                    character += chars;
                }

                at += longest.Length;
            }
        }
        finally
        {
            if (rentedBytes is not null)
            {
                ArrayPool<byte>.Shared.Return(rentedBytes);
            }

            if (rentedMatches is not null)
            {
                ArrayPool<TrieMatch>.Shared.Return(rentedMatches);
            }
        }
    }

    public TokenizerEncoding Encode(string text, string? pair, TokenizerEncodeOptions options) =>
        TokenizerEncodings.Finish(text, pair, _raw ??= Raw, PostProcessor, options, _source, this);

    public IReadOnlyList<TokenizerEncoding> EncodeBatch(IReadOnlyList<string> texts, IReadOnlyList<string>? pairs, TokenizerEncodeOptions options) =>
        TokenizerEncodings.Batch(texts, pairs, options, _source, this, _workers);

    // A text's ids with their offsets in it (no cache: a cached word has no token boundaries).
    private TokenizerEncoding Raw(string text, int sequence, int limit, bool left)
    {
        ArgumentNullException.ThrowIfNull(text);
        text = BpeTokenizer.WithoutLoneSurrogates(text);                      // one character for one: offsets still index the text
        var ids = new List<int>(text.Length / 3 + 4);
        var offsets = new List<(int Start, int End)>(ids.Capacity);
        var ranges = new List<(int Start, int Length)>();
        var starts = left && limit < int.MaxValue ? new List<(int Start, bool Added)>() : null;  // each pre-token's first token
        bool full = false;                                                    // a right limit reached: the rest is not encoded
        var pieces = t_pieces ?? new PreTokenizedText();
        t_pieces = null;
        try
        {
            void Text(int start, int length, bool atStart)
            {
                if (length == 0 || full)
                {
                    return;
                }

                _pipeline.Run(text, start, length, atStart, pieces);
                for (int i = 0; i < pieces.Pieces.Count; i++)
                {
                    var (pieceStart, pieceLength) = pieces.Pieces[i];
                    if (pieceLength == 0)
                    {
                        continue;
                    }

                    starts?.Add((ids.Count, false));

                    ranges.Clear();
                    Split(pieces.Piece(i), ids, ranges);
                    foreach (var (from, count) in ranges)
                    {
                        var (s, e) = pieces.Text.OriginOf(pieceStart + from, count);
                        offsets.Add((s, e));
                    }

                    if (starts is null && ids.Count >= limit)
                    {
                        full = true;
                        return;
                    }
                }
            }

            int last = 0;
            if (_addedPattern is not null)
            {
                foreach (var match in _addedPattern.EnumerateMatches(text))
                {
                    Text(last, match.Index - last, last == 0);
                    if (full)
                    {
                        break;
                    }

                    starts?.Add((ids.Count, true));
                    ids.Add(_vocabularySpans[text.AsSpan(match.Index, match.Length)]);
                    offsets.Add((match.Index, match.Index + match.Length));
                    last = match.Index + match.Length;
                }
            }

            Text(last, text.Length - last, last == 0);
        }
        finally
        {
            t_pieces = pieces;
        }

        return TokenizerEncodings.Raw(ids, offsets, TokenOf, sequence, starts is null ? 0 : TokenizerEncodings.Limited(starts, ids.Count, limit));
    }

    public string Decode(IEnumerable<int> ids)
    {
        ArgumentNullException.ThrowIfNull(ids);
        var tokens = new List<string>();
        foreach (int id in ids)
        {
            tokens.Add(TokenOf(id));
        }

        return _decoder is null ? string.Join(' ', tokens) : string.Concat(_decoder(tokens));
    }

    public string Decode(ReadOnlySpan<int> ids) => Decode((IEnumerable<int>)ids.ToArray());

    public JsonObject ToJson()
    {
        using var stream = new MemoryStream();
        WriteTo(stream);
        return JsonNode.Parse(stream.ToArray())!.AsObject();
    }

    public void Save(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (Directory.Exists(path))
        {
            path = Path.Combine(path, "tokenizer.json");
        }

        string? folder = Path.GetDirectoryName(Path.GetFullPath(path));
        if (folder is not null)
        {
            Directory.CreateDirectory(folder);
        }

        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 16);
        WriteTo(stream);
    }

    // The definition's JSON with the model's vocabulary written back in id order (added tokens outside it left out).
    public void WriteTo(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        var options = new JsonWriterOptions { Indented = true, Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping };
        using var writer = new Utf8JsonWriter(stream, options);
        writer.WriteStartObject();
        foreach (var (name, value) in _source)
        {
            writer.WritePropertyName(name);
            if (name != "model" || value is not JsonObject model)
            {
                if (value is null)
                {
                    writer.WriteNullValue();
                }
                else
                {
                    value.WriteTo(writer);
                }

                continue;
            }

            writer.WriteStartObject();
            foreach (var (field, fieldValue) in model)
            {
                writer.WritePropertyName(field);
                if (fieldValue is null)
                {
                    writer.WriteNullValue();
                }
                else
                {
                    fieldValue.WriteTo(writer);
                }
            }

            writer.WriteStartObject("vocab");
            foreach (var (token, id) in _vocabulary.Where(p => !_outsideModel.Contains(p.Key)).OrderBy(p => p.Value))
            {
                writer.WriteNumber(token, id);
            }

            writer.WriteEndObject();
            writer.WriteEndObject();
        }

        writer.WriteEndObject();
    }
}
