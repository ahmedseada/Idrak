// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Idrak.Abstraction.Text;
using Idrak.Models.Abstractions;

namespace Idrak.Models;

/// <summary>
/// A byte-pair-encoding tokenizer read from a Hugging Face <c>tokenizer.json</c>: byte-level BPE (GPT-2, Llama 3, Qwen,
/// …) and SentencePiece-style BPE with the "▁" word marker and byte fallback (Llama 2, Mistral, Gemma, …). Special
/// (added) tokens are matched exactly before anything else. <see cref="Encode(string)"/> adds no special tokens itself (chat
/// templates write them), like <c>add_special_tokens=False</c>. Pieces of the file it does not know are reported, not
/// guessed; other normalizer, pre-tokenizer and decoder types plug in through <see cref="TokenizerComponents"/>.
/// </summary>
public sealed class BpeTokenizer : IPretrainedTokenizer
{
    internal static readonly string Gpt2Pattern = @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
    internal static readonly char[] ByteToChar = BuildByteMap();
    private static readonly Dictionary<char, byte> CharToByte = ByteToChar.Select((c, b) => (c, b)).ToDictionary(p => p.c, p => (byte)p.b);

    private readonly Dictionary<string, int> _vocabulary;
    private readonly Dictionary<string, int>.AlternateLookup<ReadOnlySpan<char>> _vocabularySpans;
    private static readonly UTF8Encoding StrictUtf8 = new(false, throwOnInvalidBytes: true);
    // Tokenizer patterns run on every encode: compiled to IL where the runtime allows it (interpreted under native AOT).
    internal const RegexOptions Fast = RegexOptions.Compiled | RegexOptions.CultureInvariant;
    private readonly int _workers;                                            // batch threads (TokenizerOptions.Workers)
    private TokenizerEncodings.RawEncoder? _raw;                             // Raw, as Finish takes it (made once)

    // The detailed encode's span pipeline and post-processor (plan 24, T3), made when first used.
    private TokenizerPipeline? _offsetPipeline;
    private ITokenizerPostProcessor? _postProcessor;
    private bool _postProcessorRead;

    [ThreadStatic] private static PreTokenizedText? t_pieces;
    [ThreadStatic] private static List<(int Start, int Length)>? t_ranges, t_nextRanges;    // the split stages' ranges (EncodeRanges)
    [ThreadStatic] private static MergeScratch? t_merge;                                    // MergeLong's linked list and heap
    [ThreadStatic] private static List<(int Start, int Length)>? t_align;                   // a cached piece's token ranges (EncodePiece)
    private const int StackChars = 256;                                        // pieces up to this long stay on the stack

    private string[] _tokens;
    private readonly (string Left, string Right)[] _merges;
    private Dictionary<(string, string), int>? _ranks;                        // built only when merges must run on strings

    // The merges by token ids: (left id << 32 | right id) → (rank, merged id); null when some merge's parts or result
    // are not tokens (then merges run on strings).
    private readonly Dictionary<long, (int Rank, int Merged)>? _pairs;
    private readonly List<(string Content, int Id)> _added;
    private readonly HashSet<int> _special;
    private readonly HashSet<int> _addedIds;
    private readonly TokenizerRegex? _addedPattern;
    private readonly List<Func<List<string>, bool, List<string>>> _preTokenizers = [];   // (pieces, text starts the input)
    private readonly List<Func<string, string>> _normalizers = [];
    private readonly List<(string Type, Func<List<string>, List<string>> Run, Func<string, string>? Map)> _decoders = [];

    // The normalizers when each prepends a string or replaces one character by another (SentencePiece's "▁" scheme): they
    // run in place on a buffer; null when some normalizer is anything else.
    private List<(string? Prepend, char From, char To)>? _normalizerSteps = [];

    // The pre-tokenizer when it is the only one and rewrites text: Metaspace, or ByteLevel with add_prefix_space (its
    // regex, null without use_regex); these run on spans.
    private (string Replacement, string Scheme, bool Split)? _metaspace;
    private (bool Prefix, TokenizerRegex? Pattern)? _prefixByteLevel;

    // Decoding without token lists (see DecodeTable), or null; built on the first decode.
    private DecodeTable? _decodeTable;
    private readonly int _fusedDecoders = -1;                                 // decoders the table covers, -1: none

    // Pre-tokenizers that only split (no stage rewrites the text) as stages over ranges of one string, or null when
    // some stage rewrites text (then the string pipeline above runs).
    private List<(TokenizerRegex Pattern, string Behavior, bool Invert)>? _splits = [];

    // Encoded pieces (shared by concurrent encodes); looked up by span, so a hit allocates nothing. The detailed encode's
    // pieces are in the vocabulary's alphabet: they share this cache (whose entries then keep each token's range in the
    // piece, so a detailed hit aligns nothing) unless a fast path keys it by raw text whose bytes it maps first (byte
    // level), and then have their own (_alphabetCache, made when first used), counted against this one's budget: the two
    // never hold more pieces than TokenizerOptions.PieceCacheLimit together.
    private readonly PieceCache _cache;
    private PieceCache? _alphabetCache;
    private Func<int, string>? _tokenOf;                                      // TokenOf as a delegate, made once (Align takes it)

    // Byte-level decoding: the bytes of every token in one array (token id's bytes at [_byteStart[id], _byteStart[id + 1])).
    private byte[]? _tokenBytes;
    private int[]? _byteStart;

    // Byte-level encoding: the id of each byte's character (-1 when it is not a token).
    private readonly int[] _byteIds = new int[256];

    // Byte fallback: the id of each byte's <0xNN> token (-1 when it is missing).
    private readonly int[] _fallbackIds = new int[256];

    private readonly bool _byteLevel;
    private readonly bool _byteFallback;

    // The tokenizer.json this was read from (or built as), for Save; vocab and merges in it are not read again.
    private readonly JsonObject _source;

    // Added tokens that are not in the model's vocabulary (they are added to _vocabulary for lookups), so Save writes the
    // model's vocabulary as it was.
    private readonly HashSet<string> _outsideModel;
    private readonly bool _ignoreMerges;
    private readonly int? _unknown;

    private BpeTokenizer(JsonObject json, Dictionary<string, int> vocabulary, (string Left, string Right)[] merges, int cacheLimit, int workers)
    {
        var model = json["model"]?.AsObject() ?? throw new InvalidDataException("tokenizer.json has no model.");
        if ((string?)model["type"] != TokenizerModels.Bpe)
        {
            throw new NotSupportedException($"Tokenizer model '{model["type"]}' is not BPE; load it with PretrainedTokenizers.Load, "
                + "which builds every registered model type (TokenizerModels.Register adds one).");
        }

        _workers = workers;
        _vocabulary = vocabulary;
        _vocabularySpans = _vocabulary.GetAlternateLookup<ReadOnlySpan<char>>();
        _merges = merges;
        _byteFallback = (bool?)model["byte_fallback"] ?? false;
        _ignoreMerges = (bool?)model["ignore_merges"] ?? false;
        _unknown = model["unk_token"] is JsonValue unk && _vocabulary.TryGetValue((string)unk!, out int u) ? u : null;
        var addedTokens = json["added_tokens"]?.AsArray() ?? [];
        _added = [.. addedTokens.Select(t => ((string)t!["content"]!, (int)t["id"]!))];
        _special = [.. addedTokens.Where(t => (bool?)t!["special"] ?? false).Select(t => (int)t!["id"]!)];
        _addedIds = [.. _added.Select(a => a.Id)];
        _source = json;
        _outsideModel = [.. _added.Where(a => !_vocabulary.ContainsKey(a.Content)).Select(a => a.Content)];
        foreach (var (content, id) in _added)
        {
            _vocabulary[content] = id;
        }

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

        // Merged tokens are looked up by span (the two parts written side by side), so no string is built per merge.
        _pairs = new Dictionary<long, (int Rank, int Merged)>(merges.Length);
        Span<char> joined = stackalloc char[StackChars];
        for (int r = 0; r < merges.Length; r++)
        {
            var (left, right) = merges[r];
            int length = left.Length + right.Length;
            var both = length <= StackChars ? joined[..length] : new char[length];
            left.CopyTo(both);
            right.CopyTo(both[left.Length..]);
            if (!_vocabulary.TryGetValue(left, out int a) || !_vocabulary.TryGetValue(right, out int b) || !_vocabularySpans.TryGetValue(both, out int merged))
            {
                _pairs = null;
                break;
            }

            _pairs.TryAdd(((long)a << 32) | (uint)b, (r, merged));                // the first (lowest-rank) duplicate wins
        }

        if (_added.Count > 0)
        {
            _addedPattern = new TokenizerRegex(string.Join('|', _added.Select(a => a.Content).OrderByDescending(c => c.Length).Select(Regex.Escape)));
        }

        AddNormalizer(json["normalizer"]);
        _byteLevel = AddPreTokenizer(json["pre_tokenizer"]);
        if (_preTokenizers.Count != 1)
        {
            (_metaspace, _prefixByteLevel) = (null, null);
        }

        _cache = new PieceCache(cacheLimit, ranges: AlphabetKeys);

        AddDecoder(json["decoder"]);
        _fusedDecoders = FusedDecoders();
        if (_byteFallback)
        {
            for (int b = 0; b < 256; b++)
            {
                _fallbackIds[b] = _vocabulary.TryGetValue($"<0x{b:X2}>", out int id) ? id : -1;
            }
        }

        if (_byteLevel)
        {
            Span<char> one = stackalloc char[1];
            for (int b = 0; b < 256; b++)
            {
                one[0] = ByteToChar[b];
                _byteIds[b] = _vocabularySpans.TryGetValue(one, out int id) ? id : -1;
            }

            BuildTokenBytes();
        }
    }

    /// <summary>
    /// Reads tokenizer.json from a file or a model folder. The file must name the BPE model; <see cref="PretrainedTokenizers.Load"/>
    /// reads any registered model type (and other tokenizer files).
    /// </summary>
    public static BpeTokenizer Load(string path)
    {
        if (Directory.Exists(path))
        {
            path = Path.Combine(path, "tokenizer.json");
        }

        return FromDefinition(TokenizerJsonReader.ReadFile(path), new TokenizerOptions());
    }

    /// <summary>Reads a tokenizer from the JSON of a tokenizer.json.</summary>
    public static BpeTokenizer FromJson(JsonObject json)
    {
        var model = json["model"] as JsonObject;
        var vocabulary = model?["vocab"]?.AsObject().ToDictionary(p => p.Key, p => (int)p.Value!)
            ?? throw new InvalidDataException("tokenizer.json has no model vocabulary.");
        var merges = (model["merges"]?.AsArray() ?? [])
            .Select(m => m is JsonArray pair ? ((string)pair[0]!, (string)pair[1]!) : Split((string)m!)).ToArray();
        var options = new TokenizerOptions();
        return new BpeTokenizer(json, vocabulary, merges, options.ResolvedPieceCacheLimit(), options.ResolvedWorkers());
    }

    // The library's "BPE" model (TokenizerModels): the tokenizer of a definition, which it takes as its own (the added
    // tokens join its vocabulary).
    internal static BpeTokenizer FromDefinition(TokenizerDefinition definition, TokenizerOptions options) =>
        new(definition.Json,
            definition.Vocabulary ?? throw new InvalidDataException($"{definition.Source}: the BPE model has no vocabulary of token to id (model.vocab)."),
            definition.Merges ?? [], options.ResolvedPieceCacheLimit(), options.ResolvedWorkers())
        { CleanUpSpaces = options.ResolvedCleanUpSpaces(definition.CleanUpTokenizationSpaces) };

    // Whether a decode takes out the space before punctuation and contractions (TokenizerOptions.CleanUpSpaces).
    internal bool CleanUpSpaces { get; init; }

    // A tokenizer from its parts: rest is a tokenizer.json without the model's vocab and merges (the trainers build one).
    internal static BpeTokenizer FromParts(JsonObject rest, Dictionary<string, int> vocabulary, (string Left, string Right)[] merges)
        => new(rest, vocabulary, merges, new TokenizerOptions().ResolvedPieceCacheLimit(), new TokenizerOptions().ResolvedWorkers());

    // The byte-level alphabet: the character each byte is written as.
    internal static IReadOnlyList<char> ByteLevelAlphabet => ByteToChar;

    // Text as byte-level characters (its UTF-8 bytes, each as its character of the byte map).
    internal static string ByteLevelText(string text)
    {
        int count = Encoding.UTF8.GetByteCount(text);
        byte[]? rented = null;
        Span<byte> bytes = count <= 1024 ? stackalloc byte[count] : (rented = System.Buffers.ArrayPool<byte>.Shared.Rent(count));
        try
        {
            Encoding.UTF8.GetBytes(text, bytes);
            return string.Create(count, bytes[..count].ToArray(), static (chars, b) =>
            {
                for (int i = 0; i < b.Length; i++)
                {
                    chars[i] = ByteToChar[b[i]];
                }
            });
        }
        finally
        {
            if (rented is not null)
            {
                System.Buffers.ArrayPool<byte>.Shared.Return(rented);
            }
        }
    }

    /// <summary>
    /// Writes the tokenizer as a tokenizer.json (UTF-8, indented as Hugging Face writes it): everything it was read with
    /// (or trained with), the model's vocabulary in id order and the merges in rank order, so <see cref="Load"/> of the
    /// file encodes every text to the same ids.
    /// </summary>
    /// <param name="path">A file, or a folder (then its tokenizer.json).</param>
    public void Save(string path)
    {
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

    /// <summary>The tokenizer as the JSON of a tokenizer.json (see <see cref="Save"/>).</summary>
    public JsonObject ToJson()
    {
        using var stream = new MemoryStream();
        WriteTo(stream);
        return JsonNode.Parse(stream.ToArray())!.AsObject();
    }

    /// <summary>Writes the tokenizer.json (see <see cref="Save"/>) to <paramref name="stream"/>.</summary>
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
                if (field is "vocab" or "merges")
                {
                    continue;
                }

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
            writer.WriteStartArray("merges");
            foreach (var (left, right) in _merges)
            {
                writer.WriteStartArray();
                writer.WriteStringValue(left);
                writer.WriteStringValue(right);
                writer.WriteEndArray();
            }

            writer.WriteEndArray();
            writer.WriteEndObject();
        }

        writer.WriteEndObject();
    }

    /// <inheritdoc />
    public string Model => TokenizerModels.Bpe;

    /// <inheritdoc />
    public int VocabularySize => _tokens.Length;

    /// <summary>
    /// Extends the vocabulary to <paramref name="size"/> ids (models often have more embedding rows than the tokenizer has
    /// tokens, for alignment); the extra ids decode to nothing.
    /// </summary>
    public void PadVocabulary(int size)
    {
        if (size > _tokens.Length)
        {
            Array.Resize(ref _tokens, size);
        }
    }

    /// <summary>The id of <paramref name="token"/>, or null.</summary>
    public int? IdOf(string token) => _vocabulary.TryGetValue(token, out int id) ? id : null;

    /// <summary>The token with <paramref name="id"/>.</summary>
    public string TokenOf(int id) => (uint)id < (uint)_tokens.Length ? _tokens[id] ?? "" : "";

    /// <summary>Whether <paramref name="id"/> is a special token.</summary>
    public bool IsSpecial(int id) => _special.Contains(id);

    /// <inheritdoc />
    public IReadOnlyList<int> Encode(string text)
    {
        text = WithoutLoneSurrogates(text);
        var ids = new List<int>(text.Length / 3 + 4);
        EncodeSegments(text, 0, text.Length, ids);
        return ids;
    }

    /// <summary>
    /// Appends the ids of <paramref name="text"/>[<paramref name="start"/>..<paramref name="start"/> + <paramref name="length"/>]
    /// to <paramref name="ids"/>: the ids <see cref="Encode(string)"/> gives for that substring, without copying it.
    /// </summary>
    public void EncodeRange(string text, int start, int length, List<int> ids)
    {
        if (text.AsSpan(start, length).ContainsAny(Surrogates.Any))
        {
            text = WithoutLoneSurrogates(text.Substring(start, length));         // the range may cut a pair in half
            start = 0;
        }

        EncodeSegments(text, start, length, ids);
    }

    // Added tokens, and the text between them, of text[start..start + length].
    private void EncodeSegments(string text, int start, int length, List<int> ids)
    {
        int last = start;
        if (_addedPattern is not null)
        {
            foreach (var match in _addedPattern.Local.EnumerateMatches(text.AsSpan(start, length)))
            {
                int at = start + match.Index;
                EncodeText(text, last, at - last, ids, last == start);
                ids.Add(_vocabularySpans[text.AsSpan(at, match.Length)]);
                last = at + match.Length;
            }
        }

        EncodeText(text, last, start + length - last, ids, last == start);
    }

    // Half of a surrogate pair is not a character (it has no UTF-8 bytes, and Unicode normalization rejects it): it
    // becomes U+FFFD, as it does when such text is written as UTF-8.
    internal static string WithoutLoneSurrogates(string text)
    {
        int at = text.AsSpan().IndexOfAny(Surrogates.Any);
        if (at < 0)
        {
            return text;
        }

        char[]? copy = null;
        for (int i = at; i < text.Length; i++)
        {
            if (char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]))
            {
                i++;
            }
            else if (char.IsSurrogate(text[i]))
            {
                copy ??= text.ToCharArray();
                copy[i] = '\uFFFD';
            }
        }

        return copy is null ? text : new string(copy);
    }

    /// <inheritdoc />
    public TokenizerEncoding Encode(string text, string? pair, TokenizerEncodeOptions options) =>
        TokenizerEncodings.Finish(text, pair, _raw ??= Raw, PostProcessor, options, _source, this);

    /// <inheritdoc />
    public IReadOnlyList<TokenizerEncoding> EncodeBatch(IReadOnlyList<string> texts, IReadOnlyList<string>? pairs, TokenizerEncodeOptions options) =>
        TokenizerEncodings.Batch(texts, pairs, options, _source, this, _workers);

    // The post-processor the tokenizer.json names, or null.
    internal ITokenizerPostProcessor? PostProcessor
    {
        get
        {
            if (!_postProcessorRead)
            {
                _postProcessor = PostProcessors.Of(_source["post_processor"] as JsonObject);
                _postProcessorRead = true;
            }

            return _postProcessor;
        }
    }

    // A text's ids with their offsets in it, into the scratch's lists: the span pipeline (equal to this tokenizer's own,
    // plan 24 L1), each piece's ids and its tokens' ranges through the piece cache of pieces in the vocabulary's alphabet
    // (Bpe of the piece, as the merges give them, aligned to the characters they came from once, when cached). The ids are
    // the ones Encode gives. Returns how many tokens a left limit drops.
    private int Raw(string text, int limit, bool left, TokenizerEncodings.RawScratch scratch)
    {
        ArgumentNullException.ThrowIfNull(text);
        text = WithoutLoneSurrogates(text);                                  // one character for one: offsets still index the text
        var pipeline = _offsetPipeline ??= new TokenizerPipeline(_source);
        var cache = AlphabetCache;
        var tokenOf = _tokenOf ??= TokenOf;
        var (ids, offsets, ranges) = (scratch.Ids, scratch.Offsets, scratch.Ranges);
        var starts = left && limit < int.MaxValue ? scratch.Starts : null;  // each pre-token's first token
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

                pipeline.Run(text, start, length, atStart, pieces);
                for (int i = 0; i < pieces.Pieces.Count; i++)
                {
                    var (pieceStart, pieceLength) = pieces.Pieces[i];
                    if (pieceLength == 0)
                    {
                        continue;
                    }

                    starts?.Add((ids.Count, false));

                    var piece = pieces.Piece(i);
                    var normalized = pieces.Text;
                    if (cache.TryGet(piece, out var cachedIds, out var cachedRanges))
                    {
                        ids.AddRange(cachedIds);
                        foreach (int packed in cachedRanges)
                        {
                            var (from, count) = PieceCache.Range(packed);
                            offsets.Add(normalized.OriginOf(pieceStart + from, count));
                        }
                    }
                    else
                    {
                        int before = ids.Count;
                        Bpe(piece, ids);
                        var made = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(ids)[before..];
                        ranges.Clear();
                        TokenizerEncodings.Align(piece, made, tokenOf, _byteFallback, ranges);
                        cache.Add(piece, made, ranges);
                        foreach (var (from, count) in ranges)
                        {
                            offsets.Add(normalized.OriginOf(pieceStart + from, count));
                        }
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
                foreach (var match in _addedPattern.Local.EnumerateMatches(text))
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

        return starts is null ? 0 : TokenizerEncodings.Limited(starts, ids.Count, limit);
    }

    // How many pieces the caches hold together (tests: they share one PieceCacheLimit).
    internal int CachedPieces => Volatile.Read(ref _cache.Shared.Count);

    // Whether the fast paths' pieces are in the vocabulary's alphabet (no fast path maps a raw piece's bytes first), so
    // every entry of their cache is Bpe of its key.
    private bool AlphabetKeys => !_byteLevel && _prefixByteLevel is null;

    // The cache of the detailed encode's pieces (keeping each token's range): the fast paths' own when its keys are in the
    // alphabet, else one of its own on the same budget, so a raw text and an alphabet text never meet under one key and
    // the two together hold at most PieceCacheLimit pieces.
    private PieceCache AlphabetCache
    {
        get
        {
            if (_alphabetCache is { } cache)
            {
                return cache;
            }

            var made = AlphabetKeys ? _cache : new PieceCache(_cache.Shared, ranges: true);
            return Interlocked.CompareExchange(ref _alphabetCache, made, null) ?? made;
        }
    }

    /// <inheritdoc />
    public string Decode(IEnumerable<int> ids) => ids switch
    {
        int[] array => Decode(array.AsSpan()),
        List<int> list => Decode(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(list)),
        _ => Decode([.. ids]),
    };

    /// <inheritdoc />
    public string Decode(ReadOnlySpan<int> ids) => CleanUpSpaces ? DecodedSpaces.CleanUp(DecodeText(ids)) : DecodeText(ids);

    private string DecodeText(ReadOnlySpan<int> ids)
    {
        if (_byteLevel)
        {
            // Byte-level tokens are bytes written as characters (added tokens are plain text): copy each token's bytes.
            var starts = _byteStart!;
            var all = _tokenBytes!;
            int count = 0;
            foreach (int id in ids)
            {
                if ((uint)id < (uint)starts.Length - 1)
                {
                    count += starts[id + 1] - starts[id];
                }
            }

            byte[]? rented = null;
            Span<byte> bytes = count <= 1024 ? stackalloc byte[count] : (rented = System.Buffers.ArrayPool<byte>.Shared.Rent(count));
            try
            {
                int at = 0;
                foreach (int id in ids)
                {
                    if ((uint)id < (uint)starts.Length - 1)
                    {
                        var source = all.AsSpan(starts[id], starts[id + 1] - starts[id]);
                        source.CopyTo(bytes[at..]);
                        at += source.Length;
                    }
                }

                return Encoding.UTF8.GetString(bytes[..count]);
            }
            finally
            {
                if (rented is not null)
                {
                    System.Buffers.ArrayPool<byte>.Shared.Return(rented);
                }
            }
        }

        if (_fusedDecoders >= 0 && (_decodeTable ??= BuildDecodeTable()) is { Text: not null } table)
        {
            return DecodeFused(ids, table);
        }

        var tokens = new List<string>(ids.Length);
        foreach (int id in ids)
        {
            tokens.Add(TokenOf(id));
        }

        foreach (var decoder in _decoders)
        {
            tokens = decoder.Run(tokens);
        }

        return string.Concat(tokens);
    }

    // Every token's text after the decoders' Replace stages, and its byte when the ByteFallback stage reads it as one
    // (-1 otherwise); Empty* for ids without a token. Text is null when some token looks like <0xNN> but is not hex.
    private sealed record DecodeTable(string[]? Text, int[] Byte, string EmptyText, int EmptyByte);

    // How many decoders (from the first) DecodeFused reproduces: Replace stages, then at most one ByteFallback, up to the
    // first Fuse or Metaspace (from there the text is one piece, so the remaining stages run on it); -1 for other chains.
    private int FusedDecoders()
    {
        bool byteFallback = false;
        for (int i = 0; i < _decoders.Count; i++)
        {
            switch (_decoders[i].Type)
            {
                case "Replace" when !byteFallback:
                    break;
                case "ByteFallback" when !byteFallback:
                    byteFallback = true;
                    break;
                case "Fuse" or "Metaspace":
                    return i;
                default:
                    return -1;
            }
        }

        return _decoders.Count;
    }

    private DecodeTable BuildDecodeTable()
    {
        bool byteFallback = false;
        var maps = new List<Func<string, string>>();
        foreach (var (type, _, map) in _decoders.Take(_fusedDecoders))
        {
            if (map is not null)
            {
                maps.Add(map);
            }

            byteFallback |= type == "ByteFallback";
        }

        bool hex = true;
        (string, int) Entry(string token)
        {
            foreach (var map in maps)
            {
                token = map(token);
            }

            if (byteFallback && IsByteToken(token))
            {
                if (char.IsAsciiHexDigit(token[3]) && char.IsAsciiHexDigit(token[4]))
                {
                    return (token, byte.Parse(token.AsSpan(3, 2), System.Globalization.NumberStyles.AllowHexSpecifier, System.Globalization.CultureInfo.InvariantCulture));
                }

                hex = false;                                                     // left to the list stages (as they read it)
            }

            return (token, -1);
        }

        var tokens = _tokens;
        var text = new string[tokens.Length];
        var bytes = new int[tokens.Length];
        for (int id = 0; id < tokens.Length; id++)
        {
            (text[id], bytes[id]) = Entry(tokens[id] ?? "");
        }

        var (emptyText, emptyByte) = Entry("");
        return new DecodeTable(hex ? text : null, bytes, emptyText, emptyByte);
    }

    private static bool IsByteToken(string token) => token.Length == 6 && token.StartsWith("<0x", StringComparison.Ordinal) && token[5] == '>';

    // The decoders through the table: token texts copied side by side, runs of byte tokens decoded as UTF-8 (one U+FFFD
    // per byte when a run is not valid UTF-8, as the ByteFallback stage does), then the stages after the table's.
    private string DecodeFused(ReadOnlySpan<int> ids, DecodeTable table)
    {
        var texts = table.Text!;
        var tokenBytes = table.Byte;
        int most = 0;
        foreach (int id in ids)
        {
            bool known = (uint)id < (uint)texts.Length;
            most += (known ? tokenBytes[id] : table.EmptyByte) >= 0 ? 1 : (known ? texts[id] : table.EmptyText).Length;
        }

        char[]? rentedChars = null;
        byte[]? rentedBytes = null;
        Span<char> chars = most <= StackChars ? stackalloc char[StackChars] : (rentedChars = System.Buffers.ArrayPool<char>.Shared.Rent(most));
        Span<byte> run = ids.Length <= StackChars ? stackalloc byte[StackChars] : (rentedBytes = System.Buffers.ArrayPool<byte>.Shared.Rent(ids.Length));
        string text;
        try
        {
            int at = 0, pending = 0;
            foreach (int id in ids)
            {
                bool known = (uint)id < (uint)texts.Length;
                int b = known ? tokenBytes[id] : table.EmptyByte;
                if (b >= 0)
                {
                    run[pending++] = (byte)b;
                    continue;
                }

                at += DecodeRun(run[..pending], chars[at..]);
                pending = 0;
                string token = known ? texts[id] : table.EmptyText;
                token.CopyTo(chars[at..]);
                at += token.Length;
            }

            at += DecodeRun(run[..pending], chars[at..]);
            text = new(chars[..at]);
        }
        finally
        {
            if (rentedChars is not null)
            {
                System.Buffers.ArrayPool<char>.Shared.Return(rentedChars);
            }

            if (rentedBytes is not null)
            {
                System.Buffers.ArrayPool<byte>.Shared.Return(rentedBytes);
            }
        }

        List<string>? pieces = null;
        for (int i = _fusedDecoders; i < _decoders.Count; i++)
        {
            if (pieces is not null || _decoders[i].Type != "Fuse")              // fusing one piece changes nothing
            {
                pieces = _decoders[i].Run(pieces ?? [text]);
            }
        }

        return pieces is null ? text : string.Concat(pieces);
    }

    // A run of byte tokens as text: its UTF-8, or one U+FFFD per byte when it is not valid UTF-8.
    private static int DecodeRun(ReadOnlySpan<byte> run, Span<char> destination)
    {
        if (run.IsEmpty)
        {
            return 0;
        }

        if (System.Text.Unicode.Utf8.IsValid(run))
        {
            return Encoding.UTF8.GetChars(run, destination);
        }

        destination[..run.Length].Fill('\uFFFD');
        return run.Length;
    }

    // The bytes of every token (see _tokenBytes): characters of the byte map are bytes, others (and added tokens) UTF-8.
    private void BuildTokenBytes()
    {
        var starts = new int[_tokens.Length + 1];
        var bytes = new List<byte>(_tokens.Length * 8);
        Span<byte> utf8 = stackalloc byte[4];
        for (int id = 0; id < _tokens.Length; id++)
        {
            starts[id] = bytes.Count;
            string token = _tokens[id] ?? "";
            if (_addedIds.Contains(id))
            {
                bytes.AddRange(Encoding.UTF8.GetBytes(token));
                continue;
            }

            foreach (char c in token)
            {
                if (CharToByte.TryGetValue(c, out byte b))
                {
                    bytes.Add(b);
                }
                else
                {
                    int n = Encoding.UTF8.GetBytes([c], utf8);
                    bytes.AddRange(utf8[..n]);
                }
            }
        }

        starts[^1] = bytes.Count;
        _tokenBytes = [.. bytes];
        _byteStart = starts;
    }

    // Encodes text[start..start + length]. atStart: the text begins the input (Metaspace's "first" scheme only prepends
    // there, not after an added token).
    private void EncodeText(string text, int start, int length, List<int> ids, bool atStart)
    {
        if (length == 0)
        {
            return;
        }

        if (_normalizers.Count == 0)
        {
            EncodeNormalized(text.AsSpan(start, length), null, ids, atStart);
            return;
        }

        if (_normalizerSteps is { } steps)
        {
            // Prepends and character replacements in place: the text goes at the end of the buffer, each prepend before it.
            int total = length;
            foreach (var step in steps)
            {
                total += step.Prepend?.Length ?? 0;
            }

            char[]? rented = null;
            Span<char> buffer = total <= StackChars ? stackalloc char[StackChars] : (rented = System.Buffers.ArrayPool<char>.Shared.Rent(total));
            buffer = buffer[..total];
            try
            {
                int at = total - length;
                text.AsSpan(start, length).CopyTo(buffer[at..]);
                foreach (var (prepend, from, to) in steps)
                {
                    if (prepend is not null)
                    {
                        at -= prepend.Length;
                        prepend.CopyTo(buffer[at..]);
                    }
                    else
                    {
                        buffer[at..].Replace(from, to);
                    }
                }

                EncodeNormalized(buffer, null, ids, atStart);
            }
            finally
            {
                if (rented is not null)
                {
                    System.Buffers.ArrayPool<char>.Shared.Return(rented);
                }
            }

            return;
        }

        string segment = text.Substring(start, length);
        foreach (var normalize in _normalizers)
        {
            segment = normalize(segment);
        }

        EncodeNormalized(segment, segment, ids, atStart);
    }

    // Pre-tokenizes and encodes normalized text (segmentString: the same text as a string, when there is one).
    private void EncodeNormalized(ReadOnlySpan<char> segment, string? segmentString, List<int> ids, bool atStart)
    {
        if (_splits is not null)
        {
            EncodeRanges(segment, ids);
        }
        else if (_metaspace is { } metaspace)
        {
            EncodeMetaspace(segment, metaspace.Replacement, metaspace.Scheme == "always" || metaspace.Scheme == "first" && atStart, metaspace.Split, ids);
        }
        else if (_prefixByteLevel is { } byteLevel)
        {
            EncodePrefixedByteLevel(segment, byteLevel.Pattern?.Local, ids);
        }
        else
        {
            var pieces = new List<string> { segmentString ?? segment.ToString() };
            foreach (var preTokenize in _preTokenizers)
            {
                pieces = preTokenize(pieces, atStart);
            }

            foreach (var piece in pieces)
            {
                EncodePiece(piece, ids, mapBytes: false);
            }
        }
    }

    // The Metaspace pre-tokenizer on its own: spaces become the replacement, which is prepended (when the scheme says so
    // and the text does not start with it); with split, a piece starts at every replacement but the first character.
    private void EncodeMetaspace(ReadOnlySpan<char> segment, string replacement, bool prepend, bool split, List<int> ids)
    {
        int spaces = segment.Count(' ');
        int most = replacement.Length + segment.Length + spaces * (replacement.Length - 1);
        char[]? rented = null;
        Span<char> buffer = most <= StackChars ? stackalloc char[StackChars] : (rented = System.Buffers.ArrayPool<char>.Shared.Rent(most));
        try
        {
            int at = replacement.Length;
            for (var rest = segment; ;)
            {
                int space = rest.IndexOf(' ');
                if (space < 0)
                {
                    rest.CopyTo(buffer[at..]);
                    at += rest.Length;
                    break;
                }

                rest[..space].CopyTo(buffer[at..]);
                at += space;
                replacement.CopyTo(buffer[at..]);
                at += replacement.Length;
                rest = rest[(space + 1)..];
            }

            int start = replacement.Length;
            if (prepend && at > start && !buffer[start..at].StartsWith(replacement, StringComparison.Ordinal))   // empty text stays empty
            {
                start = 0;
                replacement.CopyTo(buffer);
            }

            ReadOnlySpan<char> s = buffer[start..at];
            if (!split)
            {
                EncodePiece(s, ids, mapBytes: false);
            }
            else
            {
                int from = 0;
                while (from < s.Length)
                {
                    int next = from + 1 >= s.Length ? -1 : s[(from + 1)..].IndexOf(replacement, StringComparison.Ordinal);
                    int to = next < 0 ? s.Length : from + 1 + next;
                    EncodePiece(s[from..to], ids, mapBytes: false);
                    from = to;
                }
            }
        }
        finally
        {
            if (rented is not null)
            {
                System.Buffers.ArrayPool<char>.Shared.Return(rented);
            }
        }
    }

    // The ByteLevel pre-tokenizer on its own with add_prefix_space: a space before text that does not start with one,
    // then its regex's matches (pattern; or the whole text without use_regex) encoded from their UTF-8 bytes.
    private void EncodePrefixedByteLevel(ReadOnlySpan<char> segment, Regex? pattern, List<int> ids)
    {
        char[]? rented = null;
        scoped ReadOnlySpan<char> piece = segment;
        Span<char> buffer = segment.Length < StackChars ? stackalloc char[StackChars] : default;
        try
        {
            if (!segment.StartsWith(' '))
            {
                buffer = buffer.IsEmpty ? (rented = System.Buffers.ArrayPool<char>.Shared.Rent(segment.Length + 1)) : buffer;
                buffer[0] = ' ';
                segment.CopyTo(buffer[1..]);
                piece = buffer[..(segment.Length + 1)];
            }

            if (pattern is null)
            {
                EncodePiece(piece, ids, mapBytes: true);
            }
            else
            {
                foreach (var m in pattern.EnumerateMatches(piece))
                {
                    EncodePiece(piece.Slice(m.Index, m.Length), ids, mapBytes: true);
                }
            }
        }
        finally
        {
            if (rented is not null)
            {
                System.Buffers.ArrayPool<char>.Shared.Return(rented);
            }
        }
    }

    // Splits text with the split stages (as ranges: nothing is copied, into this thread's lists) and encodes each piece.
    private void EncodeRanges(ReadOnlySpan<char> text, List<int> ids)
    {
        var splits = _splits!;
        if (splits.Count == 0)
        {
            EncodePiece(text, ids, mapBytes: _byteLevel);                     // no split stage: the text is the piece
            return;
        }

        var current = t_ranges ?? [];
        var next = t_nextRanges ?? [];
        (t_ranges, t_nextRanges) = (null, null);                               // taken (a nested call makes its own)
        try
        {
            current.Clear();
            current.Add((0, text.Length));
            foreach (var (pattern, behavior, invert) in splits)
            {
                next.Clear();
                var regex = pattern.Local;                                     // this thread's matcher, once a stage
                foreach (var range in current)
                {
                    SplitRange(text, range, regex, behavior, invert, next);
                }

                (current, next) = (next, current);
            }

            foreach (var (from, count) in current)
            {
                EncodePiece(text.Slice(from, count), ids, mapBytes: _byteLevel);
            }
        }
        finally
        {
            (t_ranges, t_nextRanges) = (current.Capacity <= 1 << 16 ? current : null, next.Capacity <= 1 << 16 ? next : null);
        }
    }

    // A pre-token piece: its cached ids, or the merges (mapBytes: the piece is raw text of a byte-level tokenizer, so its
    // UTF-8 bytes are written as the byte map's characters first). The merges run in the caller's list, after its ids, so
    // a miss allocates nothing but the cache's own entry (with each token's range in the piece, aligned in this thread's
    // list, when the cache keeps them for the detailed encode).
    private void EncodePiece(ReadOnlySpan<char> piece, List<int> ids, bool mapBytes)
    {
        if (piece.IsEmpty)
        {
            return;
        }

        if (_cache.TryGet(piece, out ReadOnlySpan<int> cached))
        {
            ids.AddRange(cached);
            return;
        }

        int before = ids.Count;
        if (mapBytes)
        {
            int most = Encoding.UTF8.GetMaxByteCount(piece.Length);
            byte[]? rented = null;
            Span<byte> bytes = most <= StackChars ? stackalloc byte[StackChars] : (rented = System.Buffers.ArrayPool<byte>.Shared.Rent(most));
            try
            {
                int count = Encoding.UTF8.GetBytes(piece, bytes);
                BpeBytes(bytes[..count], ids);
            }
            finally
            {
                if (rented is not null)
                {
                    System.Buffers.ArrayPool<byte>.Shared.Return(rented);
                }
            }
        }
        else
        {
            Bpe(piece, ids);
        }

        if (!_cache.Accepts(piece.Length))
        {
            return;
        }

        var made = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(ids)[before..];
        if (!_cache.Ranges)
        {
            _cache.Add(piece, made, null);
            return;
        }

        var ranges = t_align ??= [];
        ranges.Clear();
        TokenizerEncodings.Align(piece, made, _tokenOf ??= TokenOf, _byteFallback, ranges);
        _cache.Add(piece, made, ranges);
    }

    // Byte-level merges straight from the bytes: each byte starts as its character's token. Appends the ids to output.
    private void BpeBytes(ReadOnlySpan<byte> bytes, List<int> output)
    {
        if (_pairs is not null && !_ignoreMerges)
        {
            int before = output.Count;
            int i = 0;
            for (; i < bytes.Length && _byteIds[bytes[i]] >= 0; i++)
            {
                output.Add(_byteIds[bytes[i]]);
            }

            if (i == bytes.Length)
            {
                MergeFrom(output, before);
                return;
            }

            System.Runtime.InteropServices.CollectionsMarshal.SetCount(output, before);   // a byte without a token: the general path
        }

        // The general path, on the mapped characters.
        char[]? rentedChars = null;
        Span<char> chars = bytes.Length <= StackChars ? stackalloc char[StackChars] : (rentedChars = System.Buffers.ArrayPool<char>.Shared.Rent(bytes.Length));
        try
        {
            for (int i = 0; i < bytes.Length; i++)
            {
                chars[i] = ByteToChar[bytes[i]];
            }

            Bpe(chars[..bytes.Length], output);
        }
        finally
        {
            if (rentedChars is not null)
            {
                System.Buffers.ArrayPool<char>.Shared.Return(rentedChars);
            }
        }
    }

    // Byte-pair merges over one piece (characters of the token alphabet), best (lowest-rank) pair first, leftmost among
    // equals; appends the ids to output. Every merge joins two tokens into a token, so a character that is not one never
    // merges: the runs of known characters between such characters merge on their own, and each unknown character becomes
    // what the string merges give it (its byte fallback tokens, or the unknown token).
    private void Bpe(ReadOnlySpan<char> piece, List<int> output)
    {
        if (_ignoreMerges && _vocabularySpans.TryGetValue(piece, out int whole))
        {
            output.Add(whole);
            return;
        }

        if (_pairs is null)
        {
            BpeStrings(piece.ToString(), output);
            return;
        }

        int run = output.Count;                                               // where the current run of known characters starts
        for (int i = 0; i < piece.Length;)
        {
            int width = i + 1 < piece.Length && char.IsSurrogatePair(piece[i], piece[i + 1]) ? 2 : 1;
            if (_vocabularySpans.TryGetValue(piece.Slice(i, width), out int id))
            {
                output.Add(id);
            }
            else
            {
                MergeFrom(output, run);
                AddUnknown(piece.Slice(i, width), output);
                run = output.Count;
            }

            i += width;
        }

        MergeFrom(output, run);
    }

    // Merges output[from..] in place and drops what the merges freed.
    private void MergeFrom(List<int> output, int from)
    {
        int n = output.Count - from;
        if (n > 1)
        {
            int merged = Merge(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(output).Slice(from, n));
            System.Runtime.InteropServices.CollectionsMarshal.SetCount(output, from + merged);
        }
    }

    // A symbol that is not a token: its UTF-8 bytes' fallback tokens, or the unknown token.
    private void AddUnknown(ReadOnlySpan<char> symbol, List<int> result)
    {
        if (_byteFallback)
        {
            int most = Encoding.UTF8.GetMaxByteCount(symbol.Length);
            Span<byte> bytes = most <= StackChars ? stackalloc byte[StackChars] : new byte[most];
            foreach (byte b in bytes[..Encoding.UTF8.GetBytes(symbol, bytes)])
            {
                result.Add(_fallbackIds[b] >= 0 ? _fallbackIds[b]
                    : throw new InvalidDataException($"Byte fallback token <0x{b:X2}> is missing from the vocabulary."));
            }
        }
        else
        {
            result.Add(_unknown ?? throw new InvalidDataException($"'{symbol}' is not in the vocabulary and the tokenizer has no unknown token."));
        }
    }

    // The merges on token ids, in place; returns how many ids are left. Short pieces merge in place (lowest-rank pair,
    // leftmost among equals, repeatedly); long ones use a heap (same order, O(n log n)).
    private int Merge(Span<int> ids)
    {
        if (ids.Length > 32)
        {
            return MergeLong(ids);
        }

        int count = ids.Length;
        while (count > 1)
        {
            int best = -1, bestRank = int.MaxValue, bestMerged = 0;
            for (int i = 0; i < count - 1; i++)
            {
                if (_pairs!.TryGetValue(((long)ids[i] << 32) | (uint)ids[i + 1], out var pair) && pair.Rank < bestRank)
                {
                    (best, bestRank, bestMerged) = (i, pair.Rank, pair.Merged);
                }
            }

            if (best < 0)
            {
                break;
            }

            ids[best] = bestMerged;
            ids[(best + 2)..count].CopyTo(ids[(best + 1)..]);
            count--;
        }

        return count;
    }

    // MergeLong's working memory, one per thread, kept between pieces (a piece longer than it holds gets a larger one).
    private sealed class MergeScratch(int capacity)
    {
        public readonly int[] Symbol = new int[capacity], Next = new int[capacity], Previous = new int[capacity];
        public readonly bool[] Alive = new bool[capacity];
        public readonly PriorityQueue<(int Left, int LeftId, int RightId, int Merged), (int Rank, int Position)> Queue = new(capacity);
    }

    // The largest piece whose MergeScratch a thread keeps (longer ones are rare: their scratch is dropped after use).
    private const int KeptMergeScratch = 1 << 14;

    // Long pieces (SentencePiece-style tokenizers do not split text first): the symbols in a linked list and candidate
    // pairs in a heap ordered by (rank, position); a popped pair whose symbols changed since it was queued is skipped.
    // The result is written back to the front of ids; returns how many ids are left.
    private int MergeLong(Span<int> ids)
    {
        int n = ids.Length;
        var scratch = t_merge is { } kept && kept.Symbol.Length >= n ? kept : new MergeScratch(Math.Max(n, 256));
        t_merge = null;                                                       // taken (a nested call makes its own)
        try
        {
            var (symbol, next, previous, alive, queue) = (scratch.Symbol, scratch.Next, scratch.Previous, scratch.Alive, scratch.Queue);
            queue.Clear();
            ids.CopyTo(symbol);
            for (int i = 0; i < n; i++)
            {
                next[i] = i + 1 < n ? i + 1 : -1;
                previous[i] = i - 1;
                alive[i] = true;
            }

            void Offer(int left)
            {
                int right = left < 0 ? -1 : next[left];
                if (right >= 0 && _pairs!.TryGetValue(((long)symbol[left] << 32) | (uint)symbol[right], out var pair))
                {
                    queue.Enqueue((left, symbol[left], symbol[right], pair.Merged), (pair.Rank, left));
                }
            }

            for (int i = 0; i < n - 1; i++)
            {
                Offer(i);
            }

            int remaining = n;
            while (queue.TryDequeue(out var candidate, out _))
            {
                int left = candidate.Left, right = next[left];
                if (!alive[left] || right < 0 || symbol[left] != candidate.LeftId || symbol[right] != candidate.RightId)
                {
                    continue;
                }

                symbol[left] = candidate.Merged;
                alive[right] = false;
                remaining--;
                next[left] = next[right];
                if (next[right] >= 0)
                {
                    previous[next[right]] = left;
                }

                Offer(previous[left]);
                Offer(left);
            }

            for (int i = 0, j = 0; i >= 0; i = next[i])
            {
                ids[j++] = symbol[i];
            }

            return remaining;
        }
        finally
        {
            if (scratch.Symbol.Length <= KeptMergeScratch && (t_merge is null || t_merge.Symbol.Length < scratch.Symbol.Length))
            {
                t_merge = scratch;
            }
        }
    }

    // Merges on strings: for tokenizers whose merges name pieces that are not tokens; appends the ids to result.
    private void BpeStrings(string piece, List<int> result)
    {
        if (_ranks is null)
        {
            var ranks = new Dictionary<(string, string), int>(_merges.Length);
            for (int r = 0; r < _merges.Length; r++)
            {
                ranks.TryAdd(_merges[r], r);
            }

            Interlocked.CompareExchange(ref _ranks, ranks, null);
        }

        var symbols = new List<string>();
        for (int i = 0; i < piece.Length; i += char.IsSurrogatePair(piece, i) ? 2 : 1)
        {
            symbols.Add(char.IsSurrogatePair(piece, i) ? piece.Substring(i, 2) : piece[i].ToString());
        }

        while (symbols.Count > 1)
        {
            int best = -1, bestRank = int.MaxValue;
            for (int i = 0; i < symbols.Count - 1; i++)
            {
                if (_ranks.TryGetValue((symbols[i], symbols[i + 1]), out int rank) && rank < bestRank)
                {
                    (best, bestRank) = (i, rank);
                }
            }

            if (best < 0)
            {
                break;
            }

            symbols[best] += symbols[best + 1];
            symbols.RemoveAt(best + 1);
        }

        foreach (var symbol in symbols)
        {
            if (_vocabulary.TryGetValue(symbol, out int id))
            {
                result.Add(id);
            }
            else
            {
                AddUnknown(symbol, result);
            }
        }
    }

    // One split stage on a range: the matches and the text between them, combined as the behavior says (all adjacent,
    // so every result is again a range of the same string).
    internal static void SplitRange(ReadOnlySpan<char> text, (int Start, int Length) range, Regex pattern, string behavior, bool invert, List<(int Start, int Length)> output)
    {
        int last = range.Start, end = range.Start + range.Length;
        int pending = -1;                                                       // MergedWithNext: start of matched text waiting
        bool previousOpen = false;                                              // MergedWithPrevious: a part to extend exists
        void Add(int from, int to, bool matched)
        {
            if (to <= from)
            {
                return;
            }

            switch (behavior)
            {
                case "Isolated":
                    output.Add((from, to - from));
                    break;
                case "Removed":
                    if (!matched)
                    {
                        output.Add((from, to - from));
                    }

                    break;
                case "MergedWithPrevious":
                    if (matched && previousOpen)
                    {
                        var last = output[^1];
                        output[^1] = (last.Start, to - last.Start);
                    }
                    else
                    {
                        output.Add((from, to - from));
                    }

                    previousOpen = true;
                    break;
                case "MergedWithNext":
                    if (matched)
                    {
                        pending = pending < 0 ? from : pending;
                    }
                    else
                    {
                        int first = pending < 0 ? from : pending;
                        output.Add((first, to - first));
                        pending = -1;
                    }

                    break;
                default:
                    throw new NotSupportedException($"Split behavior '{behavior}' is not supported.");
            }
        }

        foreach (var m in pattern.EnumerateMatches(text.Slice(range.Start, range.Length)))
        {
            if (m.Length == 0)
            {
                continue;
            }

            int from = range.Start + m.Index;
            Add(last, from, invert);
            Add(from, from + m.Length, !invert);
            last = from + m.Length;
        }

        Add(last, end, invert);
        if (behavior == "MergedWithNext" && pending >= 0)
        {
            output.Add((pending, end - pending));
        }
    }

    private void AddNormalizer(JsonNode? node)
    {
        switch (node)
        {
            case null:
                return;
            case JsonObject n when TokenizerComponents.FindNormalizer((string?)n["type"]) is { } create:
                _normalizerSteps = null;                                          // a registered type: the string pipeline runs
                _normalizers.Add(create.Create(n).Normalize);
                return;
            case JsonObject n when (string?)n["type"] == "Sequence":
                foreach (var inner in n["normalizers"]!.AsArray())
                {
                    AddNormalizer(inner);
                }

                return;
            case JsonObject n:
                string type = (string)n["type"]!;
                if (type == "Prepend")
                {
                    _normalizerSteps?.Add(((string?)n["prepend"] ?? "", '\0', '\0'));
                }
                else if (type == "Replace" && n["pattern"]?["String"] is { } literal && (string)literal! is { Length: 1 } from
                         && (string)n["content"]! is { Length: 1 } to)
                {
                    _normalizerSteps?.Add((null, from[0], to[0]));
                }
                else
                {
                    _normalizerSteps = null;
                }

                _normalizers.Add(NormalizerStep(n));
                return;
        }
    }

    // A built-in normalizer type as a function; a sequence runs its normalizers in turn (registered ones too). Also the
    // library default of the type in TokenizerComponents, which an app's normalizer falls back to.
    internal static Func<string, string> NormalizerStep(JsonObject n)
    {
        string type = (string)n["type"]!;
        return type switch
        {
            "Sequence" => Chain([.. n["normalizers"]!.AsArray().OfType<JsonObject>().Select(inner =>
                TokenizerComponents.FindNormalizer((string?)inner["type"]) is { } create ? create.Create(inner).Normalize : NormalizerStep(inner))]),
            "NFC" => s => UnicodeNormalization.Normalize(s, NormalizationForm.FormC),
            "NFKC" => s => UnicodeNormalization.Normalize(s, NormalizationForm.FormKC),
            "NFD" => s => UnicodeNormalization.Normalize(s, NormalizationForm.FormD),
            "NFKD" => s => UnicodeNormalization.Normalize(s, NormalizationForm.FormKD),
            "Prepend" => s => s.Length == 0 ? s : (string)n["prepend"]! + s,   // empty text stays empty, as in Hugging Face's
            "Replace" => Replacer(n),
            "Lowercase" => BertComponents.Lowercase,
            "Strip" => Stripper((bool?)n["strip_left"] ?? true, (bool?)n["strip_right"] ?? true),
            "BertNormalizer" => BertComponents.StringNormalizer(n),
            "Precompiled" => PrecompiledCharsMap.Step(n),                         // the charsmap read once, here
            _ => throw new NotSupportedException($"Tokenizer normalizer '{type}' is not supported; add it with TokenizerComponents.RegisterNormalizer."),
        };

        static Func<string, string> Chain(Func<string, string>[] steps) => s =>
        {
            foreach (var step in steps)
            {
                s = step(s);
            }

            return s;
        };
    }

    private static Func<string, string> Stripper(bool left, bool right) => s => (left, right) switch
    {
        (true, true) => s.Trim(),
        (true, false) => s.TrimStart(),
        (false, true) => s.TrimEnd(),
        _ => s,
    };

    private static Func<string, string> Replacer(JsonObject n)
    {
        string content = (string)n["content"]!;
        if (n["pattern"]?["String"] is { } literal)
        {
            string from = (string)literal!;
            return s => s.Replace(from, content, StringComparison.Ordinal);
        }

        var regex = new TokenizerRegex((string)n["pattern"]!["Regex"]!);
        return s => regex.Local.Replace(s, content);
    }

    // Returns true when the tokenizer is byte-level (its pieces are bytes written as characters).
    private bool AddPreTokenizer(JsonNode? node)
    {
        if (node is not JsonObject p)
        {
            return false;
        }

        string type = (string)p["type"]!;
        if (TokenizerComponents.FindPreTokenizer(type) is { } create)
        {
            _preTokenizers.Add(Stage(create.Create(p)));
            _splits = null;                                                       // a registered type: the string pipeline runs
            return false;
        }

        switch (type)
        {
            case "Sequence":
                bool byteLevel = false;
                foreach (var inner in p["pretokenizers"]!.AsArray())
                {
                    byteLevel |= AddPreTokenizer(inner);
                }

                return byteLevel;
            case "Split":
                var (pattern, behavior, invert) = SplitSettings(p);
                _preTokenizers.Add((pieces, atStart) => SplitPieces(pieces, pattern, behavior, invert));
                _splits?.Add((pattern, behavior, invert));
                return false;
            case "ByteLevel":
                bool prefix = (bool?)p["add_prefix_space"] ?? false, useRegex = (bool?)p["use_regex"] ?? true;
                var gpt2 = new TokenizerRegex(Gpt2Pattern);
                if (prefix)
                {
                    _splits = null;                                               // rewrites text: the string pipeline runs
                }
                else if (useRegex)
                {
                    _splits?.Add((gpt2, "Isolated", false));
                }

                _prefixByteLevel = prefix ? (true, useRegex ? gpt2 : null) : null;
                _preTokenizers.Add(ByteLevelStage(prefix, useRegex, gpt2));
                return true;
            case "Metaspace":
                var (replacement, scheme, split) = MetaspaceSettings(p);
                _splits = null;
                _metaspace = (replacement, scheme, split);
                _preTokenizers.Add(MetaspaceStage(replacement, scheme, split));
                return false;
            case "Digits":
                var digits = DigitsPattern(p);
                _preTokenizers.Add((pieces, atStart) => SplitPieces(pieces, digits, "Isolated", false));
                _splits?.Add((digits, "Isolated", false));
                return false;
            case "Whitespace":
                var words = new TokenizerRegex(WordsPattern);
                _preTokenizers.Add(WhitespaceStage(words));
                _splits?.Add((words, "Removed", true));                           // keeps the matches only
                return false;
            case "WhitespaceSplit":
                var runs = new TokenizerRegex(NonWhitespacePattern);
                _preTokenizers.Add((pieces, atStart) => SplitPieces(pieces, runs, "Removed", true));
                _splits?.Add((runs, "Removed", true));
                return false;
            case "Punctuation":
                var (punctuation, punctuationBehavior) = PunctuationSettings(p);
                _preTokenizers.Add((pieces, atStart) => SplitPieces(pieces, punctuation, punctuationBehavior, false));
                _splits?.Add((punctuation, punctuationBehavior, false));
                return false;
            case "BertPreTokenizer":
                var whitespaceRuns = new TokenizerRegex(NonWhitespacePattern);
                var (bertPunctuation, _) = PunctuationSettings(p);
                _preTokenizers.Add(BertComponents.StringPreTokenizer());
                _splits?.Add((whitespaceRuns, "Removed", true));
                _splits?.Add((bertPunctuation, "Isolated", false));
                return false;
            default:
                throw new NotSupportedException($"Tokenizer pre-tokenizer '{type}' is not supported; add it with TokenizerComponents.RegisterPreTokenizer.");
        }
    }

    // A registered pre-tokenizer as a stage of the string pipeline.
    private static Func<List<string>, bool, List<string>> Stage(IPreTokenizer part) =>
        (pieces, atStart) => part.PreTokenize(pieces, atStart) is var result && result is List<string> list ? list : [.. result];

    // A built-in pre-tokenizer type as a stage of the string pipeline; a sequence runs its pre-tokenizers in turn
    // (registered ones too). Also the library default of the type in TokenizerComponents, which an app's falls back to.
    internal static Func<List<string>, bool, List<string>> PreTokenizerStage(JsonObject p)
    {
        string type = (string)p["type"]!;
        switch (type)
        {
            case "Sequence":
                Func<List<string>, bool, List<string>>[] stages = [.. p["pretokenizers"]!.AsArray().OfType<JsonObject>().Select(inner =>
                    TokenizerComponents.FindPreTokenizer((string?)inner["type"]) is { } create ? Stage(create.Create(inner)) : PreTokenizerStage(inner))];
                return (pieces, atStart) =>
                {
                    foreach (var stage in stages)
                    {
                        pieces = stage(pieces, atStart);
                    }

                    return pieces;
                };
            case "Split":
                var (pattern, behavior, invert) = SplitSettings(p);
                return (pieces, atStart) => SplitPieces(pieces, pattern, behavior, invert);
            case "ByteLevel":
                return ByteLevelStage((bool?)p["add_prefix_space"] ?? false, (bool?)p["use_regex"] ?? true, new TokenizerRegex(Gpt2Pattern));
            case "Metaspace":
                var (replacement, scheme, split) = MetaspaceSettings(p);
                return MetaspaceStage(replacement, scheme, split);
            case "Digits":
                var digits = DigitsPattern(p);
                return (pieces, atStart) => SplitPieces(pieces, digits, "Isolated", false);
            case "Whitespace":
                return WhitespaceStage(new TokenizerRegex(WordsPattern));
            case "WhitespaceSplit":
                var runs = new TokenizerRegex(NonWhitespacePattern);
                return (pieces, atStart) => SplitPieces(pieces, runs, "Removed", true);
            case "Punctuation":
                var (punctuation, behavior2) = PunctuationSettings(p);
                return (pieces, atStart) => SplitPieces(pieces, punctuation, behavior2, false);
            case "BertPreTokenizer":
                return BertComponents.StringPreTokenizer();
            default:
                throw new NotSupportedException($"Tokenizer pre-tokenizer '{type}' is not supported; add it with TokenizerComponents.RegisterPreTokenizer.");
        }
    }

    internal const string WordsPattern = @"\w+|[^\w\s]+";

    // WhitespaceSplit keeps the runs between whitespace; Punctuation splits at each punctuation character (Unicode's P
    // categories and the ASCII punctuation Unicode files as symbols, as Rust's is_ascii_punctuation counts them).
    internal const string NonWhitespacePattern = @"\S+";
    internal const string PunctuationPattern = @"[\p{P}$+<=>^`|~]";

    internal static (TokenizerRegex Pattern, string Behavior, bool Invert) SplitSettings(JsonObject p) =>
        (new TokenizerRegex(p["pattern"]!["Regex"] is { } r ? (string)r! : Regex.Escape((string)p["pattern"]!["String"]!)),
         (string?)p["behavior"] ?? "Isolated", (bool?)p["invert"] ?? false);

    internal static (string Replacement, string Scheme, bool Split) MetaspaceSettings(JsonObject p) =>
        ((string?)p["replacement"] ?? "▁", (string?)p["prepend_scheme"] ?? (((bool?)p["add_prefix_space"] ?? true) ? "always" : "never"),
         (bool?)p["split"] ?? true);

    internal static (TokenizerRegex Pattern, string Behavior) PunctuationSettings(JsonObject p) =>
        (new TokenizerRegex(PunctuationPattern), (string?)p["behavior"] ?? "Isolated");

    internal static TokenizerRegex DigitsPattern(JsonObject p) => new(((bool?)p["individual_digits"] ?? false) ? @"\p{Nd}" : @"\p{Nd}+");

    private static Func<List<string>, bool, List<string>> ByteLevelStage(bool prefix, bool useRegex, TokenizerRegex pattern) => (pieces, atStart) =>
    {
        var result = new List<string>(pieces.Count);
        var gpt2 = pattern.Local;
        foreach (var text in pieces)
        {
            string piece = prefix && !text.StartsWith(' ') ? " " + text : text;
            if (!useRegex)
            {
                result.Add(ToByteChars(piece));
                continue;
            }

            foreach (var m in gpt2.EnumerateMatches(piece))
            {
                result.Add(ToByteChars(piece.AsSpan(m.Index, m.Length)));
            }
        }

        return result;
    };

    private static Func<List<string>, bool, List<string>> MetaspaceStage(string replacement, string scheme, bool split) => (pieces, atStart) =>
    {
        var result = new List<string>();
        for (int i = 0; i < pieces.Count; i++)
        {
            string s = pieces[i].Replace(" ", replacement, StringComparison.Ordinal);
            if ((scheme == "always" || scheme == "first" && i == 0 && atStart) && s.Length > 0 && !s.StartsWith(replacement, StringComparison.Ordinal))
            {
                s = replacement + s;
            }

            if (!split)
            {
                result.Add(s);
                continue;
            }

            int start = 0;
            for (int j = 1; j <= s.Length; j++)
            {
                if (j == s.Length || s.AsSpan(j).StartsWith(replacement, StringComparison.Ordinal))
                {
                    result.Add(s[start..j]);
                    start = j;
                }
            }
        }

        return result;
    };

    private static Func<List<string>, bool, List<string>> WhitespaceStage(TokenizerRegex pattern) => (pieces, atStart) =>
    {
        var result = new List<string>(pieces.Count);
        var words = pattern.Local;
        foreach (var piece in pieces)
        {
            foreach (var m in words.EnumerateMatches(piece))
            {
                result.Add(piece.Substring(m.Index, m.Length));
            }
        }

        return result;
    };

    // A split stage on pieces of the string pipeline: SplitRange on each piece, then the ranges copied out.
    private static List<string> SplitPieces(List<string> pieces, TokenizerRegex pattern, string behavior, bool invert)
    {
        var result = new List<string>(pieces.Count);
        var ranges = new List<(int Start, int Length)>();
        var regex = pattern.Local;
        foreach (var piece in pieces)
        {
            if (behavior is not ("Isolated" or "Removed" or "MergedWithPrevious" or "MergedWithNext"))
            {
                throw new NotSupportedException($"Split behavior '{behavior}' is not supported.");
            }

            ranges.Clear();
            SplitRange(piece, (0, piece.Length), regex, behavior, invert, ranges);
            foreach (var (start, length) in ranges)
            {
                result.Add(length == piece.Length ? piece : piece.Substring(start, length));
            }
        }

        return result;
    }

    // Text as the byte map's characters of its UTF-8 bytes.
    private static string ToByteChars(ReadOnlySpan<char> text)
    {
        int count = Encoding.UTF8.GetByteCount(text);
        byte[]? rented = null;
        char[]? rentedChars = null;
        Span<byte> bytes = count <= StackChars ? stackalloc byte[StackChars] : (rented = System.Buffers.ArrayPool<byte>.Shared.Rent(count));
        Span<char> chars = count <= StackChars ? stackalloc char[StackChars] : (rentedChars = System.Buffers.ArrayPool<char>.Shared.Rent(count));
        try
        {
            Encoding.UTF8.GetBytes(text, bytes);
            for (int i = 0; i < count; i++)
            {
                chars[i] = ByteToChar[bytes[i]];
            }

            return new string(chars[..count]);
        }
        finally
        {
            if (rented is not null)
            {
                System.Buffers.ArrayPool<byte>.Shared.Return(rented);
                System.Buffers.ArrayPool<char>.Shared.Return(rentedChars!);
            }
        }
    }

    private void AddDecoder(JsonNode? node)
    {
        if (node is not JsonObject d)
        {
            return;
        }

        string type = (string)d["type"]!;
        if (TokenizerComponents.FindDecoder(type) is { } create)
        {
            // A registered type: named so that no built-in stage (or the fused decoding) takes it for one of its own.
            _decoders.Add(("Registered:" + type, Stage(create.Create(d)), null));
            return;
        }

        switch (type)
        {
            case "Sequence":
                foreach (var inner in d["decoders"]!.AsArray())
                {
                    AddDecoder(inner);
                }

                break;
            case "ByteLevel":
                break;                                                           // handled in Decode
            case "Replace":
                var replace = Replacer(d);
                _decoders.Add((type, tokens => [.. tokens.Select(replace)], replace));
                break;
            default:
                _decoders.Add((type, DecoderStage(d), null));                    // ByteFallback, Fuse, Strip, Metaspace
                break;
        }
    }

    // A registered decoder as a stage.
    private static Func<List<string>, List<string>> Stage(ITokenizerDecoder part) =>
        tokens => part.Decode(tokens) is var result && result is List<string> list ? list : [.. result];

    // A built-in decoder type as a stage; a sequence runs its decoders in turn (registered ones too). Also the library
    // default of the type in TokenizerComponents, which an app's decoder falls back to; there ByteLevel turns the byte
    // characters back into text (a byte-level tokenizer does it in Decode, from its table of token bytes).
    internal static Func<List<string>, List<string>> DecoderStage(JsonObject d)
    {
        string type = (string)d["type"]!;
        switch (type)
        {
            case "Sequence":
                Func<List<string>, List<string>>[] stages = [.. d["decoders"]!.AsArray().OfType<JsonObject>().Select(inner =>
                    TokenizerComponents.FindDecoder((string?)inner["type"]) is { } create ? Stage(create.Create(inner)) : DecoderStage(inner))];
                return tokens =>
                {
                    foreach (var stage in stages)
                    {
                        tokens = stage(tokens);
                    }

                    return tokens;
                };
            case "ByteLevel":
                return tokens => [ByteLevelText(tokens)];
            case "Replace":
                var replace = Replacer(d);
                return tokens => [.. tokens.Select(replace)];
            case "ByteFallback":
                return ByteFallback;
            case "Fuse":
                return tokens => [string.Concat(tokens)];
            case "Strip":
                string content = (string)d["content"]!;
                int start = (int?)d["start"] ?? 0, stop = (int?)d["stop"] ?? 0;
                return tokens => Strip(tokens, content, start, stop);
            case "WordPiece":
                return BertComponents.WordPieceDecoder(d);
            case "Metaspace":
                string replacement = (string?)d["replacement"] ?? "▁";
                return tokens =>
                {
                    var text = string.Concat(tokens).Replace(replacement, " ", StringComparison.Ordinal);
                    return [text.StartsWith(' ') ? text[1..] : text];
                };
            default:
                throw new NotSupportedException($"Tokenizer decoder '{type}' is not supported; add it with TokenizerComponents.RegisterDecoder.");
        }
    }

    // Tokens of byte characters (the GPT-2 byte map) as the UTF-8 text of their bytes; other characters as they are.
    private static string ByteLevelText(List<string> tokens)
    {
        var bytes = new List<byte>();
        Span<byte> utf8 = stackalloc byte[4];
        foreach (var token in tokens)
        {
            foreach (char c in token)
            {
                if (CharToByte.TryGetValue(c, out byte b))
                {
                    bytes.Add(b);
                }
                else
                {
                    int n = Encoding.UTF8.GetBytes([c], utf8);
                    bytes.AddRange(utf8[..n]);
                }
            }
        }

        return Encoding.UTF8.GetString(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(bytes));
    }

    // Byte tokens (<0xNN>) in a row become their UTF-8 text.
    private static List<string> ByteFallback(List<string> tokens)
    {
        var result = new List<string>();
        var bytes = new List<byte>();
        void Flush()
        {
            if (bytes.Count > 0)
            {
                // As the tokenizers library: bytes that are not valid UTF-8 become one U+FFFD each.
                try
                {
                    result.Add(StrictUtf8.GetString(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(bytes)));
                }
                catch (DecoderFallbackException)
                {
                    result.Add(new string('�', bytes.Count));
                }

                bytes.Clear();
            }
        }

        foreach (var token in tokens)
        {
            if (IsByteToken(token))
            {
                bytes.Add(char.IsAsciiHexDigit(token[3]) && char.IsAsciiHexDigit(token[4])
                    ? byte.Parse(token.AsSpan(3, 2), System.Globalization.NumberStyles.AllowHexSpecifier, System.Globalization.CultureInfo.InvariantCulture)
                    : Convert.ToByte(token[3..5], 16));
            }
            else
            {
                Flush();
                result.Add(token);
            }
        }

        Flush();
        return result;
    }

    private static List<string> Strip(List<string> tokens, string content, int start, int stop)
    {
        if (tokens.Count == 0)
        {
            return tokens;
        }

        string first = tokens[0];
        for (int i = 0; i < start && first.StartsWith(content, StringComparison.Ordinal); i++)
        {
            first = first[content.Length..];
        }

        tokens[0] = first;
        string last = tokens[^1];
        for (int i = 0; i < stop && last.EndsWith(content, StringComparison.Ordinal); i++)
        {
            last = last[..^content.Length];
        }

        tokens[^1] = last;
        return tokens;
    }

    internal static (string, string) Split(string merge)
    {
        int space = merge.IndexOf(' ', 1);
        return (merge[..space], merge[(space + 1)..]);
    }

    // GPT-2's byte → printable character table: printable bytes map to themselves, the rest to 256 + n.
    private static char[] BuildByteMap()
    {
        var map = new char[256];
        var printable = Enumerable.Range('!', '~' - '!' + 1).Concat(Enumerable.Range('¡', '¬' - '¡' + 1)).Concat(Enumerable.Range('®', 'ÿ' - '®' + 1)).ToHashSet();
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            map[b] = printable.Contains(b) ? (char)b : (char)(256 + n++);
        }

        return map;
    }
}
